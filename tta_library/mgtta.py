from copy import deepcopy
import torch
import torch.nn as nn
import torch.jit
from models.vpt import PromptViT
import functools
import os
from quant_library.quant_layers.matmul import *
from tta_library.metanet.TTT import TTTMGG, TTTConfig
RUNNING_IMAGNET_R = False
#foa loss
class MGTTA(nn.Module):
    """test-time Forward Only Adaptation
    FOA devises both input level and output level adaptation.
    It avoids modification to model weights and adapts in a backpropogation-free manner.
    """
    def __init__(self, model:PromptViT, mgg:nn.Module, adapt_lr=1e-3, beta1=0.9, beta2=0.99, fitness_lambda=0.4, norm_dim=768):
        super().__init__()

        self.model = model
        self.mgg = mgg

        self.fitness_lambda = fitness_lambda
        self.beta1 = beta1
        self.beta2 = beta2
        self.hist_stat = None # which is used for calculating the shift direction in Eqn. (8)
        self.norm_dim = norm_dim

        self.model_state, self.mgg_state = copy_model_and_mgg(self.model, self.mgg)

        self.norm_param_names, _ = collect_norm_params(self.model)
        
        self.adapt_lr = adapt_lr
        
        self.hook_handle = []
        self.init_state()

    def init_state(self):
        self.grad_m = {}
        self.grad_v = {}

        for name, p in self.model.named_parameters(): 
            if name in self.norm_param_names:
                self.grad_m[name] = torch.zeros_like(p, requires_grad=False)
                self.grad_v[name] = torch.zeros_like(p, requires_grad=False)

        self.last_mini_batch_params_dict = None        

        self.result_params_list = []
        self.batch_cnt = 0
        
    def _update_hist(self, batch_mean):
        """Update overall test statistics, Eqn. (9)"""
        if self.hist_stat is None:
            self.hist_stat = batch_mean
        else:
            self.hist_stat = 0.9 * self.hist_stat + 0.1 * batch_mean
            
    def _get_shift_vector(self):
        """Calculate shift direction, Eqn. (8)"""
        if self.hist_stat is None:
            return None
        else:
            return self.train_info[1][-768:] - self.hist_stat
        
    def forward(self, x):
        self.batch_cnt += 1
        outputs, loss, _ = forward_and_get_loss(x, self.model, self.imagenet_mask, self.fitness_lambda, self.train_info)

        # get gradient
        loss.backward()

        result_params = {}
    
        # Feed the optimizee into the optimizer to compute parameter updates
        mgg_inputs_list = []
        for (name, p) in self.model.named_parameters():
            if name in self.norm_param_names:
                m, v = self.grad_m[name], self.grad_v[name]
                grad = p.grad.data
                m.mul_(self.beta1).add_((1 - self.beta1) * grad)
                self.grad_m[name] = m
                mt_hat = m / (1-self.beta1**(self.batch_cnt + 1))
                v.mul_(self.beta2).add_((1 - self.beta2) * (grad ** 2))
                self.grad_v[name] = v
                vt_hat = v / (1-self.beta2**(self.batch_cnt + 1))
                mt_tilde = mt_hat / (torch.sqrt(vt_hat) + 1e-8)
                gt_tilde = grad / (torch.sqrt(vt_hat) + 1e-8)

                mt_tilde = mt_tilde.view(self.norm_dim, 1, -1) 
                gt_tilde = gt_tilde.view(self.norm_dim, 1, -1)

                mgg_inputs = torch.cat([mt_tilde, gt_tilde], dim=2) 
                
                mgg_inputs_list.append(mgg_inputs)
                
        mgg_inputs = torch.cat(mgg_inputs_list, dim=0)
        with torch.no_grad():
            updates, last_mini_batch_params_dict = self.mgg(mgg_inputs, last_mini_batch_params_dict=self.last_mini_batch_params_dict)
        self.last_mini_batch_params_dict = last_mini_batch_params_dict
        
        # update each norm params
        update_idx=0
        for (name, p) in self.model.named_parameters():
            if name in self.norm_param_names:
                updates_i = updates[update_idx:update_idx+self.norm_dim]
                result_params[name] = p.detach().clone() - updates_i.view(*p.size()) * self.adapt_lr
                update_idx += self.norm_dim
                
        for name in result_params:
            rsetattr(self.model, name, nn.Parameter(result_params[name].clone().detach()))

        return outputs.detach(), loss.detach()
    
    def obtain_origin_stat(self, train_loader, train_info_path=None):
        if train_info_path is not None and os.path.exists(train_info_path):
            self.train_info = torch.load(train_info_path)
        else:
            print('===> begin calculating mean and variance')
            features = []
            with torch.no_grad():
                for _, dl in enumerate(train_loader):
                    images = dl[0].cuda()
                    feature = self.model.layers_cls_features(images)
                    features.append(feature)
                features = torch.cat(features, dim=0)
                self.train_info = torch.std_mean(features, dim=0)

            # preparing quantized model for prompt adaptation
            for _, m in self.model.vit.named_modules():
                if type(m) == PTQSLBatchingQuantMatMul:
                    m._get_padding_parameters(torch.zeros((1,12,197+self.model.num_prompts,64)).cuda(), torch.zeros((1,12,64,197+self.model.num_prompts)).cuda())
                elif type(m) == SoSPTQSLBatchingQuantMatMul:
                    m._get_padding_parameters(torch.zeros((1,12,197+self.model.num_prompts,197+self.model.num_prompts)).cuda(), torch.zeros((1,12,197+self.model.num_prompts,64)).cuda())
            print('===> calculating mean and variance end')

            # torch.save(self.train_info)

    def get_mgg_ckpt(self):
        mgg_ckpt_dict = {
                        'model_state_dict': self.mgg.state_dict(),
                    }
        return mgg_ckpt_dict
    
    def get_vit_ckpt(self):
        vit_ckpt_dict = {
                        'model_state_dict': self.model.state_dict(),
                    }
        return vit_ckpt_dict

    def reset(self):
        #mgg reset
        load_model_and_mgg(self.model, self.mgg, self.model_state, self.mgg_state)
        
        self.init_state()
        
    def configure_model(self):
        """Configure model for use with tent."""
        # train mode, because tent optimizes the model to minimize entropy
        self.model.train()
        # disable grad, to (re-)enable only what tent updates
        self.model.requires_grad_(False)
        # configure norm for tent updates: enable grad + force batch statisics
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad_(True)
                # force use of batch stats in train and eval modes
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
            if isinstance(m, (nn.GroupNorm, nn.LayerNorm)):
                m.requires_grad_(True)
        # configure mgg for updates: enable grad
        self.mgg.requires_grad_(True)
        
def copy_model_and_mgg(model, mgg):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    mgg_state = deepcopy(mgg.state_dict())
    return model_state, mgg_state

def load_model_and_mgg(model, mgg, model_state, mgg_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    mgg.load_state_dict(mgg_state)

@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    temprature = 1
    x = x/ temprature
    x = -(x.softmax(1) * x.log_softmax(1)).sum(1)
    return x

criterion_mse = nn.MSELoss(reduction='none').cuda()

def forward_and_get_loss(images, model:PromptViT, imagenet_mask, fitness_lambda, train_info):
    features = model.layers_cls_features(images)
    """discrepancy loss for Eqn. (5)"""
    batch_std, batch_mean = torch.std_mean(features, dim=0)
    std_mse, mean_mse = criterion_mse(batch_std, train_info[0]), criterion_mse(batch_mean, train_info[1])
    # NOTE: $lambda$ should be 0.2 for ImageNet-R!!
    discrepancy_loss = fitness_lambda * (std_mse.sum() + mean_mse.sum()) / 64
        
    cls_features = features[:, -768:] # the feature of classification token
    output = model.vit.head(cls_features)

    """entropy loss for Eqn. (5)"""
    if imagenet_mask is not None:
        output = output[:, imagenet_mask]

    # loss = F.cross_entropy(output, target, reduction='mean')
    entropy_loss = softmax_entropy(output).mean() 
    loss = discrepancy_loss + entropy_loss
                
    cls_features_mean = cls_features.mean(dim=0)
    return output, loss, cls_features_mean
    

    
def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)

# using wonder's beautiful simplification: https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-objects/31174427?noredirect=1#comment86638618_31174427

def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))

def collect_norm_params(model):
    """Collect the affine scale + shift parameters from batch norms.
    Walk the model's modules and collect all batch normalization parameters.
    Return the parameters and their names.
    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    params_num = 0
    for nm, m in model.named_modules():
        if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np}")
                    params_num += p.nelement()
    print(f'params_num={params_num}')
    return names, params 




def create_mgg(model_path, hidden_size=768, num_attention_heads=1):
    input_dim = 2 
    output_dim = 1
    configuration = TTTConfig(hidden_size=hidden_size, num_attention_heads=num_attention_heads, mini_batch_size=1,)
    ttt_mgg = TTTMGG(configuration, 0, input_dim=input_dim, output_dim=output_dim)
    
    # load model
    ckpt = torch.load(model_path)
    ttt_mgg.load_state_dict(ckpt['model_state_dict'])
    print(ttt_mgg)
    return ttt_mgg


