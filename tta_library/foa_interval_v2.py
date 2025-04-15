"""
Copyright to FOA Authors ICML 2024
"""

from argparse import ArgumentDefaultsHelpFormatter
from copy import deepcopy

import torch
import torch.nn as nn
import torch.jit

from torch.autograd import Variable
from models.vpt import PromptViT
import cma
import numpy as np
import time
import math

from utils.cli_utils import accuracy, AverageMeter
from calibration_library.metrics import ECELoss
from queue import PriorityQueue
from quant_library.quant_layers.matmul import *

RUNNING_IMAGNET_R = False

class CMA_Collect_Images(nn.Module):
    """
    FOA variants which stores images for interval update
    In implementation, we simulate BS=1 by forwarding each sample one by one
    """
    def __init__(self, model:PromptViT, fitness_lambda=0.4):
        super().__init__()

        self.model = model
        self.fitness_lambda = fitness_lambda
        self.es = self._init_cma() # initialization for CMA-ES

        self.best_prompts = model.prompts
        self.best_loss = np.inf

        self.hist_stat = None # which is used for calculating the shift direction in Eqn. (8)

    def _init_cma(self):
        dim = self.model.prompts.numel()
        popsize = 27 # which is equal to 4 + 3 * np.log(dim) when #prompts=3
        cma_opts = {
            'seed': 2020,
            'popsize': popsize,
            'maxiter': -1,
            'verbose': -1,
        }
        es = cma.CMAEvolutionStrategy(dim * [0], 1, inopts=cma_opts)
        self.popsize = es.popsize
        return es

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
        """calculating shift direction, Eqn. (8)"""
        shift_vector = self._get_shift_vector()

        self.best_loss, self.best_outputs, batch_means = np.inf, None, []

        """Sampling from CMA-ES and evaluate the new solutions.
        Note that we also compare the current solutions with the previous best one"""
        prompts, losses = self.es.ask() + [self.best_prompts.flatten().cpu()], []
        for j, prompt in enumerate(prompts):
            self.model.prompts = torch.nn.Parameter(torch.tensor(prompt, dtype=torch.float).
                                                        reshape_as(self.model.prompts).cuda())
            self.model.prompts.requires_grad_(False)

            outputs, loss, batch_mean = forward_and_get_loss(x, self.model, self.fitness_lambda, self.train_info, shift_vector, self.imagenet_mask)
            batch_means.append(batch_mean[-768:].unsqueeze(0))
            del batch_mean

            if self.best_loss > loss.item():
                self.best_prompts = self.model.prompts
                self.best_loss = loss.item()
                self.best_outputs = outputs
                outputs = None
            losses.append(loss.item())
            del outputs

            print(f'Solution:[{j+1}/{len(prompts)}], Loss: {loss.item()}')

        """CMA-ES updates, Eqn. (6)"""
        self.es.tell(prompts, losses)
        
        """Update overall test statistics, Eqn. (9)"""
        batch_means = torch.cat(batch_means, dim=0).mean(0)
        self._update_hist(batch_means)
        return self.best_outputs
    
    def obtain_origin_stat(self, train_loader):
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
        for n, m in self.model.vit.named_modules():
            if type(m) == PTQSLBatchingQuantMatMul:
                m._get_padding_parameters(torch.zeros((1,12,197+self.model.num_tokens,64)).cuda(), torch.zeros((1,12,64,197+self.model.num_tokens)).cuda())
            elif type(m) == SoSPTQSLBatchingQuantMatMul:
                m._get_padding_parameters(torch.zeros((1,12,197+self.model.num_tokens,197+self.model.num_tokens)).cuda(), torch.zeros((1,12,197+self.model.num_tokens,64)).cuda())
        print('===> calculating mean and variance end')

    def reset(self):
        self.es = self._init_cma()
        self.hist_stat = None

        self.model.reset()
        self.best_prompts = self.model.prompts

@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    temprature = 1
    x = x/ temprature
    x = -(x.softmax(1) * x.log_softmax(1)).sum(1)
    return x

criterion_mse = nn.MSELoss(reduction='none')

def forward_and_get_loss(images, model:PromptViT, fitness_lambda, train_info, shift_vector, imagenet_mask):
    features, outputs = [], []
    for image in images:
        feature = model.layers_cls_features_with_prompts(image.unsqueeze(0))
        features.append(feature)
        cls_features = feature[:, -768:]
        output = model.vit.head(cls_features)
        if imagenet_mask is not None:
            output = output[:, imagenet_mask]

        if shift_vector is not None:
            output = model.vit.head(cls_features + 1. * shift_vector)
            if imagenet_mask is not None:
                output = output[:, imagenet_mask]
        outputs.append(output)

    features = torch.cat(features, dim=0)
    outputs = torch.cat(outputs, dim=0)

    batch_std, batch_mean = torch.std_mean(features, dim=0)
    std_mse, mean_mse = criterion_mse(batch_std, train_info[0]), criterion_mse(batch_mean, train_info[1])
    # NOTE: $lambda$ should be 0.2 for ImageNet-R!!
    discrepancy_loss = fitness_lambda * (std_mse.sum() + mean_mse.sum()) * images.shape[0] / 64
    
    entropy_loss = softmax_entropy(output).sum()
    loss = discrepancy_loss + entropy_loss
    return outputs, loss, batch_mean