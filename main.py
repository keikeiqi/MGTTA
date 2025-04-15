import os
import time
import argparse
import random
import math
from importlib import reload, import_module

from utils.utils import get_logger
from utils.cli_utils import *
from dataset.selectedRotateImageFolder import prepare_test_data
from dataset.ImageNetMask import imagenet_r_mask
from dataset.ImageNetMask import imagenet_a_mask
import torch    
import torch.nn.functional as F

import timm
import numpy as np

import tta_library.tent as tent
import tta_library.eata as eata
import tta_library.deyo as deyo
import tta_library.sar as sar
import tta_library.cotta as cotta
import tta_library.foa_bp as foa_bp
import tta_library.mgtta_lstm as mgtta_lstm
import tta_library.mgtta as mgtta
import tta_library.foa_sgd as foa_sgd
import tta_library.train_mgg as train_mgg
from tta_library.sam import SAM
from tta_library.t3a import T3A
from tta_library.foa import FOA
from tta_library.foa_shift import Shift
from tta_library.lame import LAME
from tta_library.foa_interval_v2 import CMA_Collect_Images
from tta_library.foa_interval_v1 import CMA_Collect_Features

from calibration_library.metrics import ECELoss

from quant_library.quant_utils.models import get_net
from quant_library.quant_utils import net_wrap
import quant_library.quant_utils.datasets as datasets
from quant_library.quant_utils.quant_calib import HessianQuantCalibrator

from models.vpt import PromptViT, FOAViT
import os.path as osp

def validate(val_loader, model, args):
    acc_list = []
    
    batch_time = AverageMeter('Time', ':6.3f')
    adapt_time = AverageMeter('Adapt time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, top1, top5],
        prefix='Test: ')
    
    outputs_list, targets_list = [], []
    with torch.no_grad():
        end = time.time()
        for i, dl in enumerate(val_loader):
            load_data_end = time.time()#新加
            images, target = dl[0], dl[1]
            if args.gpu is not None:
                images = images.cuda()
            if torch.cuda.is_available():
                target = target.cuda()

            torch.cuda.reset_peak_memory_stats()
            if args.algorithm == 'tent_bp_gtloss':
                output = model(images, target)#需要gt算gtloss                
            else:
                output = model(images)
            print(f'memory usage: {torch.cuda.max_memory_allocated()/(1024*1024):.3f}MB')

            # for calculating Expected Calibration Error (ECE)
            outputs_list.append(output.cpu())
            targets_list.append(target.cpu())

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            del output

            # measure elapsed time
            batch_time.update(time.time() - end)
            adapt_time.update(time.time() - load_data_end)
            end = time.time()
            if i % 5 == 0:
                logger.info(progress.display(i))
                
            acc_list.append(top1.avg)      
        outputs_list = torch.cat(outputs_list, dim=0).numpy()
        targets_list = torch.cat(targets_list, dim=0).numpy()
        
        logits = args.algorithm != 'lame' # only lame outputs probability
        ece_avg = ECELoss().loss(outputs_list, targets_list, logits=logits) # calculate ECE
        
    args.acc_record[args.corruption] = acc_list       

    return top1.avg, top5.avg, ece_avg

def validate_mgtta(val_loader, model, args, show_eval_msg=True):
    # 记录当前corruption的acc变化
    acc_list = []
    
    batch_time = AverageMeter('Time', ':6.3f')
    adapt_time = AverageMeter('Adapt time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    loss_record = AverageMeter('Loss', ':6.3f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, top1, top5, loss_record],
        prefix='Test: ')
    
    outputs_list, targets_list = [], []

    # with torch.no_grad(): 

    end = time.time()
    for i, dl in enumerate(val_loader):
        load_data_end = time.time()#
        images, target = dl[0], dl[1]
        # print(target)
        # torch.cuda.reset_peak_memory_stats()
        
        if args.gpu is not None:
            images = images.cuda()
        if torch.cuda.is_available():
            target = target.cuda()

        output, loss = model(images)
        # print(f'memory usage: {torch.cuda.max_memory_allocated()/(1024*1024):.3f}MB')
            
        # for calculating Expected Calibration Error (ECE)
        outputs_list.append(output.cpu())
        targets_list.append(target.cpu())

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))
        loss_record.update(loss.item(), images.size(0))
        
        del output

        # measure elapsed time
        batch_time.update(time.time() - end)
        adapt_time.update(time.time() - load_data_end)#新加
        end = time.time()
        if i % 5 == 0 and show_eval_msg:
            logger.info(progress.display(i))
            
        acc_list.append(top1.avg)
        # print(top1.avg)
        
        
    outputs_list = torch.cat(outputs_list, dim=0).numpy()
    targets_list = torch.cat(targets_list, dim=0).numpy()

    logits = args.algorithm != 'lame' # only lame outputs probability
    ece_avg = ECELoss().loss(outputs_list, targets_list, logits=logits) # calculate ECE

    args.acc_record[args.corruption] = acc_list
    
    return top1.avg, top5.avg, ece_avg

def end2end_train_mgg_update_normlayer_eval(val_loader, model, args):
    best_eval_acc=0
    best_ep=0

    for ep in range(args.train_mgg_epoch):
        # init
        logger.info('---------------------')
        logger.info(f'epoch={ep}')
        model.init_state()
        
        batch_time = AverageMeter('Time', ':6.3f')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        loss_record = AverageMeter('Loss', ':6.3f')
        progress = ProgressMeter(
            len(val_loader),
            [batch_time, top1, top5, loss_record],
            prefix='Test: ')
        
        outputs_list, targets_list = [], []

        end = time.time()
        for i, dl in enumerate(val_loader):
            images, target = dl[0], dl[1]
            if args.gpu is not None:
                images = images.cuda()
            if torch.cuda.is_available():
                target = target.cuda()
            output, loss = model(images, target)

            # for calculating Expected Calibration Error (ECE)
            outputs_list.append(output.cpu())
            targets_list.append(target.cpu())

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))
            loss_record.update(loss, images.size(0))
            del output

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if i % 5 == 0:
                logger.info(progress.display(i))

            if torch.isnan(loss) or torch.isinf(loss):
                logger.info(f'loss={loss}')
                logger.info('error. loss is nan or inf, break!')
                raise Exception('loss is nan or inf, break!')

        outputs_list = torch.cat(outputs_list, dim=0).numpy()
        targets_list = torch.cat(targets_list, dim=0).numpy()

        logits = args.algorithm != 'lame' # only lame outputs probability
        ece_avg = ECELoss().loss(outputs_list, targets_list, logits=logits) # calculate ECE
    
        # eval MGG on ImageNet-C validation set
        if ep % args.eval_interval_epochs == 0:
            #save MGG for each epoch
            mgg_ckpt_dict = model.get_mgg_ckpt()
            torch.save(mgg_ckpt_dict, osp.join(args.output, f'MGG_ckpt_epoch{ep}.pth'))
            
            logger.info(f'evaluation for epoch {ep}:')
            eval_mgg = model.get_mgg_clone()
            eval_corrupt_acc, eval_corrupt_ece = [], []

            # create model for eval
            eval_net = timm.create_model('vit_base_patch16_224', pretrained=True).cuda()
            eval_net = PromptViT(eval_net, num_prompts=0).cuda()
        
            if args.algorithm == 'train_mgg':
                eval_adapt_model = mgtta.MGTTA(eval_net, eval_mgg, adapt_lr=args.eval_adapt_lr, norm_dim=args.norm_dim)
            else:
                raise NotImplementedError
            _, train_loader = obtain_train_loader(args) 
            eval_adapt_model.obtain_origin_stat(train_loader, args.train_info_path)
            eval_adapt_model.configure_model() 
            
            eval_corruptions = ['speckle_noise', 'spatter', 'gaussian_blur', 'saturate']
            args.batch_size = 64
            for corrupt in eval_corruptions:
                args.corruption = corrupt
                eval_adapt_model.imagenet_mask = None
                eval_dataset, eval_loader = prepare_test_data(args)
                eval_top1, eval_top5, eval_ece_loss = validate_mgtta(eval_loader, eval_adapt_model, args, show_eval_msg=True)
                logger.info(f"Eval under shift type {args.corruption} After {args.algorithm} Top-1 Accuracy: {eval_top1:.6f} and Top-5 Accuracy: {eval_top5:.6f} and ECE: {eval_ece_loss:.6f}")
                eval_corrupt_acc.append(eval_top1)
                eval_corrupt_ece.append(eval_ece_loss)
                eval_adapt_model.reset()

            logger.info(f'eval corrupt type list: {eval_corruptions}')
            logger.info(f'eval mean acc of corruption: {sum(eval_corrupt_acc)/len(eval_corrupt_acc) if len(eval_corrupt_acc) else 0}')
            # logger.info(f'eval mean ece of corruption: {sum(eval_corrupt_ece)/len(eval_corrupt_ece) if len(eval_corrupt_ece) else 0}')
            logger.info(f'eval corrupt acc list: {[_.item() for _ in eval_corrupt_acc]}')
            # logger.info(f'eval corrupt ece list: {[_*100 for _ in eval_corrupt_ece]}')
            
            eval_mean_acc = sum(eval_corrupt_acc)/len(eval_corrupt_acc)
            if eval_mean_acc > best_eval_acc:
                # save the best MGG
                best_eval_acc = eval_mean_acc
                mgg_ckpt_dict = model.get_mgg_ckpt()
                torch.save(mgg_ckpt_dict, osp.join(args.output, f'best_MGG_ckpt.pth'))
                best_ep=ep
            logger.info(f'best eval acc: {best_eval_acc} at epoch {best_ep}')
            
    return top1.avg, top5.avg, ece_avg


def obtain_train_loader(args):
    args.corruption = 'original'
    train_dataset, train_loader = prepare_test_data(args)
    train_dataset.switch_mode(True, False)
    return train_dataset, train_loader

def init_config(config_name):
    """initialize the config. Use reload to make sure it's fresh one!"""
    _,_,files =  next(os.walk("./quant_lib/configs"))
    if config_name+".py" in files:
        quant_cfg = import_module(f"quant_lib.configs.{config_name}")
    else:
        raise NotImplementedError(f"Invalid config name {config_name}")
    reload(quant_cfg)
    return quant_cfg

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_args():

    parser = argparse.ArgumentParser(description='PyTorch ImageNet-C Testing')

    # path of data, output dir
    parser.add_argument('--data', default='/data/imagenet', help='path to dataset')
    parser.add_argument('--data_sketch', default='/data/imagenet-sketch', help='path to dataset')
    parser.add_argument('--data_adv', default='/dataimagenet-a/', help='path to dataset')
    parser.add_argument('--data_v2', default='/data/imagenet-v2/', help='path to dataset')
    parser.add_argument('--data_corruption', default='/data/imagenet-c', help='path to dataset')
    parser.add_argument('--data_rendition', default='/data/imagenet-r', help='path to dataset')

    # general parameters, dataloader parameters
    parser.add_argument('--seed', default=2020, type=int, help='seed for initializing training. ')
    parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')
    parser.add_argument('--debug', default=False, type=bool, help='debug or not.')
    parser.add_argument('--workers', default=2, type=int, help='number of data loading workers (default: 4)')
    parser.add_argument('--batch_size', default=64, type=int, help='mini-batch size (default: 64)')
    parser.add_argument('--if_shuffle', default=True, type=str2bool, help='if shuffle the test set.')

    # algorithm selection
    parser.add_argument('--algorithm', default='train_mgg', type=str, help='supporting foa, sar, cotta and etc.')

    # dataset settings
    parser.add_argument('--level', default=5, type=int, help='corruption level of test(val) set.')
    parser.add_argument('--corruption', default='gaussian_noise', type=str, help='corruption type of test(val) set.')
    parser.add_argument('--dataset', default='imagenet_c_test', type=str, choices=['imagenet_c_test', 'imagenet_r', 'imagenet_sketch', 'imagenet_a', 'gaussian_noise', 'imagenet_c_val_mix', 'imagenet_c_val'], help='dataset name')
    parser.add_argument('--reset_seed_for_each_corruption',default=True, type=str2bool, help='reset seed for each corruption')
    parser.add_argument('--used_data_num', default=-1, type=int, help='number of used data for mgg training')
    
    # model settings
    parser.add_argument('--quant', default=False, action='store_true', help='whether to use quantized model in the experiment')

    # output settings
    parser.add_argument('--output', default='./outputs', help='the output directory of this experiment')
    parser.add_argument('--tag', default='experiment_tag', type=str, help='the tag of experiment')


    # eata settings
    parser.add_argument('--fisher_size', default=2000, type=int, help='number of samples to compute fisher information matrix.')
    parser.add_argument('--fisher_alpha', type=float, default=2000., help='the trade-off between entropy and regularization loss, in Eqn. (8)')
    parser.add_argument('--e_margin', type=float, default=math.log(1000)*0.40, help='entropy margin E_0 in Eqn. (3) for filtering reliable samples')
    parser.add_argument('--d_margin', type=float, default=0.05, help='\epsilon in Eqn. (5) for filtering redundant samples')
    parser.add_argument('--exp_type', default='continual', type=str, help='continual or each_shift_reset') 

    # sar setting    
    parser.add_argument('--sar_margin_e0', default=0.4*math.log(1000), type=float, help='the entropy margin for sar')    

    # DeYO setting
    parser.add_argument('--patch_len', default=4, type=int, help="patch")
    parser.add_argument('--ent_threshold', default=0.5, type=float)
    parser.add_argument('--deyo_ent_threshold', default=0.4, type=float)
    parser.add_argument('--div_threshold', default=0.05, type=float)
    parser.add_argument('--reset_constant', default=0.005, type=float, help="threshold e_m for model recovery scheme")
    parser.add_argument('--aug_type', default='patch', type=str, help='patch, pixel, occ')
    parser.add_argument('--occlusion_size', default=112, type=int)
    parser.add_argument('--row_start', default=56, type=int)
    parser.add_argument('--column_start', default=56, type=int)
    parser.add_argument('--deyo_margin', default=0.5, type=float, help='Entropy threshold for sample selection $\tau_\mathrm{Ent}$ in Eqn. (8)')
    parser.add_argument('--deyo_margin_e0', default=0.4, type=float, help='Entropy margin for sample weighting $\mathrm{Ent}_0$ in Eqn. (10)')
    parser.add_argument('--plpd_threshold', default=0.2, type=float, help='PLPD threshold for sample selection $\tau_\mathrm{PLPD}$ in Eqn. (8)')
    parser.add_argument('--filter_ent', default=1, type=int)
    parser.add_argument('--filter_plpd', default=1, type=int)
    parser.add_argument('--reweight_ent', default=1, type=int)
    parser.add_argument('--reweight_plpd', default=1, type=int)
    parser.add_argument('--wandb_interval', default=100, type=int, help='print outputs to wandb at given interval.')
    parser.add_argument('--wandb_log', default=0, type=int)

    # FOA settings
    parser.add_argument('--num_prompts', default=0, type=int, help='number of inserted prompts for test-time adaptation.')    
    parser.add_argument('--fitness_lambda', default=0.4, type=float, help='the balance factor $lambda$ for Eqn. (5) in FOA')    
    parser.add_argument('--lambda_bp', default=30, type=float, help='the balance factor $lambda$ for Eqn. (5) in FOA-BP')    

    # train MGG
    parser.add_argument('--train_info_path', default='./train_info.pt', type=str, help='path of training statistics')
    parser.add_argument('--eval_adapt_lr', default=1e-3, type=float, help="lr for TTA updates during evaluation")    
    parser.add_argument('--train_mgg_lr', default=1e-2, type=float, help='lr rate for MGG during training')
    parser.add_argument('--train_adapt_lr', default=1e-4, type=float, help='lr rate for norm layer during training')
    parser.add_argument('--train_mgg_epoch', default=40, type=int, help='Number of epochs for training MGG')
    parser.add_argument('--eval_interval_epochs', default=1, type=int, help='How many epochs to interval between each evaluation?')
    
    # MGTTA
    parser.add_argument('--adapt_lr', default=0.005, type=float, help='lr for TTA methods')
    parser.add_argument('--mgg_path', default='', type=str, help='ckpt path of mgg')
    # ttt
    parser.add_argument('--norm_dim', default=768, type=int, help='Dimension of norm layer for ViT')
    parser.add_argument('--ttt_hidden_size', default=8, type=int, help='TTT hidden size')
    parser.add_argument('--num_attention_heads', default=1, type=int, help='num_attention_heads')    
    # lstm 
    parser.add_argument('--lstm_hidden_sz', default=8, type=int, help='Number of hidden layer neurons in LSTM')
    parser.add_argument('--unroll_len', default=10, type=int, help='unroll length for training LSTM')


    return parser.parse_args()



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "set_deterministic"):
        torch.set_deterministic(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    args = get_args()

    # set random seeds
    if args.seed is not None:
        set_seed(args.seed)

    # create logger for experiment
    args.output += '/' + args.algorithm + args.tag + '/'
    if not os.path.exists(args.output):
        os.makedirs(args.output, exist_ok=True)
    
    logger = get_logger(name="project", output_directory=args.output, log_name=time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())+"-log.txt", debug=False)
    logger.info(args)

    # configure the domains for adaptation
    # options: ['imagenet_c_test', 'imagenet_r', 'imagenet_sketch', 'imagenet_a', 'gaussian_noise', 'imagenet_c_val_mix', 'imagenet_c_val']
    if args.dataset == 'imagenet_c_test':
        corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog', 'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']
    elif args.dataset == 'imagenet_r':
        corruptions = ['rendition']
    elif args.dataset == 'imagenet_sketch':
        corruptions = ['sketch']
    elif args.dataset == 'imagenet_a':
        corruptions = ['imagenet_a']
    elif args.dataset == 'gaussian_noise':
        corruptions = ['gaussian_noise']
    elif args.dataset == 'imagenet_c_val':
        corruptions = ['speckle_noise', 'spatter', 'gaussian_blur', 'saturate']
    elif args.dataset == 'imagenet_c_val_mix': # for training MGG
        corruptions= ['imagenet_c_val_mix']
    else:
        raise NotImplementedError
    print("corruptions:", corruptions)
    
    # create model
    if args.quant:
        # Use PTQ4Vit for model quantization
        # NOTE the bit of quantization can be modified in quant_lib/configs/PTQ4ViT.py
        quant_cfg = init_config("PTQ4ViT")
        net = get_net('vit_base_patch16_224')
        wrapped_modules = net_wrap.wrap_modules_in_net(net,quant_cfg)
        g=datasets.ViTImageNetLoaderGenerator(args.data,'imagenet',32,32,16,kwargs={"model":net})
        test_loader=g.test_loader()
        calib_loader=g.calib_loader(num=32)
        
        quant_calibrator = HessianQuantCalibrator(net,wrapped_modules,calib_loader,sequential=False,batch_size=4) # 16 is too big for ViT-L-16
        quant_calibrator.batching_quant_calib()
    else:
        # full precision model
        net = timm.create_model('vit_base_patch16_224', pretrained=True)
        
    net = net.cuda()
    net.eval()
    net.requires_grad_(False)

    if args.algorithm == 'tent':
        net = tent.configure_model(net)
        params, _ = tent.collect_params(net)
        optimizer = torch.optim.SGD(params, args.adapt_lr, momentum=0.9)
        adapt_model = tent.Tent(net, optimizer)
    elif args.algorithm == 'eata':
        # compute fisher informatrix
        args.corruption = 'original'
        fisher_dataset, fisher_loader = prepare_test_data(args)
        fisher_dataset.set_dataset_size(args.fisher_size)
        fisher_dataset.switch_mode(True, False)

        net = eata.configure_model(net)
        params, param_names = eata.collect_params(net)
        ewc_optimizer = torch.optim.SGD(params, 0.001)
        fishers = {}
        train_loss_fn = nn.CrossEntropyLoss().cuda()
        for iter_, (images, targets) in enumerate(fisher_loader, start=1):      
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                targets = targets.cuda(args.gpu, non_blocking=True)
            outputs = net(images)
            _, targets = outputs.max(1)
            loss = train_loss_fn(outputs, targets)
            loss.backward()
            for name, param in net.named_parameters():
                if param.grad is not None:
                    if iter_ > 1:
                        fisher = param.grad.data.clone().detach() ** 2 + fishers[name][0]
                    else:
                        fisher = param.grad.data.clone().detach() ** 2
                    if iter_ == len(fisher_loader):
                        fisher = fisher / iter_
                    fishers.update({name: [fisher, param.data.clone().detach()]})
            ewc_optimizer.zero_grad()
        logger.info("compute fisher matrices finished")
        del ewc_optimizer

        optimizer = torch.optim.SGD(params, args.adapt_lr, momentum=0.9)
        adapt_model = eata.EATA(net, optimizer, fishers, args.fisher_alpha, e_margin=args.e_margin, d_margin=args.d_margin)

    elif args.algorithm == 'deyo':
        net = deyo.configure_model(net)
        params, _ = deyo.collect_params(net)
        optimizer = torch.optim.SGD(params, args.adapt_lr, momentum=0.9)
        adapt_model = deyo.DeYO(net, args, optimizer, deyo_margin=args.deyo_margin, margin_e0=args.deyo_margin_e0)
    elif args.algorithm == 'foa':
        net = PromptViT(net, args.num_prompts).cuda()
        adapt_model = FOA(net, args.fitness_lambda)
        _, train_loader = obtain_train_loader(args)
        adapt_model.obtain_origin_stat(train_loader, args.train_info_path)  
    elif args.algorithm == 'foa_shift':
        # activation shifting doesn't need to insert prompts 
        net = PromptViT(net, 0).cuda()
        adapt_model = Shift(net)
        _, train_loader = obtain_train_loader(args)
        adapt_model.obtain_origin_stat(train_loader)
    elif args.algorithm == 'foaI_v1':
        net = PromptViT(net, args.num_prompts).cuda()
        adapt_model = CMA_Collect_Features(net, args.fitness_lambda)
        _, train_loader = obtain_train_loader(args)
        adapt_model.obtain_origin_stat(train_loader)
    elif args.algorithm == 'foaI_v2':
        net = PromptViT(net, args.num_prompts).cuda()
        adapt_model = CMA_Collect_Images(net, args.fitness_lambda)
        _, train_loader = obtain_train_loader(args)
        adapt_model.obtain_origin_stat(train_loader)
    elif args.algorithm == 'foa_bp':
        # foa_bp updates the normalization layers, thus no prompt is needed
        net = PromptViT(net, 0).cuda()
        net = foa_bp.configure_model(net).cuda()
        params, _ = foa_bp.collect_params(net)
        optimizer = torch.optim.SGD(params, 0.005, momentum=0.9)
        adapt_model = foa_bp.FOA_BP(net, optimizer, args.lambda_bp)
        _, train_loader = obtain_train_loader(args)
        adapt_model.obtain_origin_stat(train_loader)
    elif args.algorithm == 't3a':
        # NOTE: set num_classes to 200 on ImageNet-R
        adapt_model = T3A(net, 1000, 20).cuda()
    elif args.algorithm == 'sar':
        net = sar.configure_model(net)
        params, _ = sar.collect_params(net)
        base_optimizer = torch.optim.SGD
        optimizer = SAM(params, base_optimizer, lr=args.adapt_lr, momentum=0.9)
        # NOTE: set margin_e0 to 0.4*math.log(200) on ImageNet-R
        adapt_model = sar.SAR(net, optimizer, margin_e0=args.sar_margin_e0)
    elif args.algorithm == 'cotta':
        net = cotta.configure_model(net)
        params, _ = cotta.collect_params(net)
        optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9)
        adapt_model = cotta.CoTTA(net, optimizer, steps=1, episodic=False)
    elif args.algorithm == 'lame':
        adapt_model = LAME(net)
    elif args.algorithm == 'foa_sgd':
        net = PromptViT(net, num_prompts=0).cuda()
        params, _ = foa_sgd.collect_params(net)
        optimizer = torch.optim.SGD(params, args.adapt_lr)
        adapt_model = foa_sgd.FOA_SGD(net, optimizer)
        _, train_loader = obtain_train_loader(args) 
        adapt_model.obtain_origin_stat(train_loader, args.train_info_path)           
        adapt_model.configure_model() 
    elif args.algorithm == 'train_mgg':
        net = FOAViT(net).cuda()
        mgg = train_mgg.create_mgg(args.ttt_hidden_size, args.num_attention_heads).cuda()
        optimizer = torch.optim.Adam(mgg.parameters(), args.train_mgg_lr)
        adapt_model = train_mgg.TrainMGG(net, mgg, optimizer, args.train_adapt_lr, norm_dim=args.norm_dim)
        _, train_loader = obtain_train_loader(args)
        adapt_model.obtain_origin_stat(train_loader, args.train_info_path)       
        adapt_model.configure_model()
    elif args.algorithm == 'mgtta':
        net = FOAViT(net).cuda()
        mgg = mgtta.create_mgg(args.mgg_path, hidden_size=args.ttt_hidden_size, num_attention_heads=args.num_attention_heads).cuda()
        adapt_model = mgtta.MGTTA(net, mgg, args.adapt_lr, norm_dim=args.norm_dim)
        _, train_loader = obtain_train_loader(args) 
        adapt_model.obtain_origin_stat(train_loader, args.train_info_path)
        adapt_model.configure_model()  
    # elif args.algorithm == 'mgtta_lstm':
    #     net = PromptViT(net, num_prompts=0).cuda()
    #     mgg = mgtta_lstm.create_mgg(args.mgg_path, args.lstm_hidden_sz).cuda()
    #     adapt_model = mgtta_lstm.MGTTA_LSTM(net, mgg, args.adapt_lr)
    #     _, train_loader = obtain_train_loader(args) 
    #     adapt_model.obtain_origin_stat(train_loader, args.train_info_path)         
    #     adapt_model.configure_model()      
    else:
        assert False, NotImplementedError
 

    corrupt_acc, corrupt_ece = [], []
    args.acc_record = {}
    for corrupt in corruptions:
        # reset the seed for each corruptions
        if args.reset_seed_for_each_corruption:
            set_seed(args.seed)

        args.corruption = corrupt
        logger.info(args.corruption)

        if args.corruption == 'rendition':
            adapt_model.imagenet_mask = imagenet_r_mask
        elif args.corruption == 'imagenet_a':
            adapt_model.imagenet_mask = imagenet_a_mask
        else:
            adapt_model.imagenet_mask = None

        val_dataset, val_loader = prepare_test_data(args)

        if args.algorithm == 'train_mgg':
            top1, top5, ece_loss = end2end_train_mgg_update_normlayer_eval(val_loader, adapt_model, args)
        elif 'mgtta' in args.algorithm: 
            top1, top5, ece_loss = validate_mgtta(val_loader, adapt_model, args)
        else:
            top1, top5, ece_loss = validate(val_loader, adapt_model, args)

        logger.info(f"Under shift type {args.corruption} After {args.algorithm} Top-1 Accuracy: {top1:.6f} and Top-5 Accuracy: {top5:.6f} and ECE: {ece_loss:.6f}")
        corrupt_acc.append(top1)
        corrupt_ece.append(ece_loss)

        # reset model before adapting on the next domain
        adapt_model.reset()
        
    logger.info(f'mean acc of corruption: {sum(corrupt_acc)/len(corrupt_acc) if len(corrupt_acc) else 0}')
    logger.info(f'mean ece of corruption: {sum(corrupt_ece)/len(corrupt_ece) if len(corrupt_ece) else 0}')
    logger.info(f'corrupt type list: {corruptions}')
    logger.info(f'corrupt acc list: {[_.item() for _ in corrupt_acc]}')
    logger.info(f'corrupt ece list: {[_*100 for _ in corrupt_ece]}')