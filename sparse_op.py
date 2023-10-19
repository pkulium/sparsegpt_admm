import torch
from torch import autograd, nn
import torch.nn.functional as F

from itertools import repeat
import collections.abc as container_abcs


class Sparse(autograd.Function):
    """" Prune the unimprotant weight for the forwards phase but pass the gradient to dense weight using SR-STE in the backwards phase"""

    @staticmethod
    def forward(ctx, weight, N, M, decay = 0.0002):
        ctx.save_for_backward(weight)

        output = weight.clone()
        length = weight.numel()
        group = int(length/M)

        weight_temp = weight.detach().abs().reshape(group, M)
        index = torch.argsort(weight_temp, dim=1)[:, :int(M-N)]

        w_b = torch.ones(weight_temp.shape, device=weight_temp.device)
        w_b = w_b.scatter_(dim=1, index=index, value=0).reshape(weight.shape)
        ctx.mask = w_b
        ctx.decay = decay

        return output*w_b


    @staticmethod
    def backward(ctx, grad_output):

        weight, = ctx.saved_tensors
        return grad_output + ctx.decay * (1-ctx.mask) * weight, None, None
class Sparse_NHWC(autograd.Function):
    """" Prune the unimprotant edges for the forwards phase but pass the gradient to dense weight using SR-STE in the backwards phase"""

    @staticmethod
    def forward(ctx, weight, N, M, decay = 0.0002):

        ctx.save_for_backward(weight)
        output = weight.clone()
        length = weight.numel()
        group = int(length/M)

        weight_temp = weight.detach().abs().permute(0,2,3,1).reshape(group, M)
        index = torch.argsort(weight_temp, dim=1)[:, :int(M-N)]

        w_b = torch.ones(weight_temp.shape, device=weight_temp.device)
        w_b = w_b.scatter_(dim=1, index=index, value=0).reshape(weight.permute(0,2,3,1).shape)
        w_b = w_b.permute(0,3,1,2)

        ctx.mask = w_b
        ctx.decay = decay

        return output*w_b

    @staticmethod
    def backward(ctx, grad_output):

        weight, = ctx.saved_tensors
        return grad_output + ctx.decay * (1-ctx.mask) * weight, None, None

class SparseConv(nn.Conv2d):
    """" implement N:M sparse convolution layer """
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', N=2, M=4, **kwargs):
        self.N = N
        self.M = M
        super(SparseConv, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, **kwargs)


    def get_sparse_weights(self):

        return Sparse_NHWC.apply(self.weight, self.N, self.M)



    def forward(self, x):

        w = self.get_sparse_weights()
        x = F.conv2d(
            x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return x



class SparseLinear(nn.Linear):

    def __init__(self, in_features: int, out_features: int, bias: bool = True, N=2, M=2, decay = 0.0002, **kwargs):
        self.N = N
        self.M = M
        super(SparseLinear, self).__init__(in_features, out_features, bias = True)


    def get_sparse_weights(self):

        return Sparse.apply(self.weight, self.N, self.M)



    def forward(self, x):

        w = self.get_sparse_weights()
        x = F.linear(x, w, self.bias)
        return x


    

import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import math

class VRPGELinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scores = nn.Parameter(torch.Tensor(self.weight.size()[0], 1))
        self.register_buffer('subnet', torch.zeros_like(self.scores))
        self.train_weights = False
        nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))
        self.prune = True
        self.subnet = torch.ones_like(self.scores)
        self.register_buffer("stored_mask_0", torch.zeros_like(self.scores))
        self.register_buffer("stored_mask_1", torch.zeros_like(self.scores))

    @property
    def clamped_scores(self):
        return self.scores

    def fix_subnet(self):
        self.subnet = (torch.rand_like(self.scores) < self.clamped_scores).float()

    def forward(self, x):
        if self.prune:
            if not self.train_weights:
                self.subnet = StraightThroughBinomialSampleNoGrad.apply(self.scores)
                self.stored_mask_0.data = (self.subnet-self.scores)/torch.sqrt((self.scores+1e-20)*(1-self.scores+1e-20))
                w = self.weight * self.subnet.view(-1, 1)
                x = F.linear(x, w, self.bias)
            else:
                w = self.weight * self.subnet.view(-1, 1)
                x = F.linear(x, w, self.bias)
        else:
            x = F.linear(x, self.weight, self.bias)
        return x

class StraightThroughBinomialSampleNoGrad(autograd.Function):
    @staticmethod
    def forward(ctx, scores):
        output = (torch.rand_like(scores) < scores).float()
        return output

    @staticmethod
    def backward(ctx, grad_outputs):
        return torch.zeros_like(grad_outputs)


import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ProbMaskLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))  # Probability
        self.subnet = None  # Mask
        self.train_weights = False
        nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))
        self.T = 1

    @property
    def clamped_scores(self):
        return self.scores

    def fix_subnet(self):
        self.subnet = (torch.rand_like(self.scores) < self.clamped_scores).float()

    def forward(self, x):
        if not self.train_weights:  # training
            eps = 1e-20
            temp = self.T
            uniform0 = torch.rand_like(self.scores)
            uniform1 = torch.rand_like(self.scores)
            noise = -torch.log(torch.log(uniform0 + eps) / torch.log(uniform1 + eps) + eps)
            self.subnet = torch.sigmoid((torch.log(self.clamped_scores + eps) - torch.log(1.0 - self.clamped_scores + eps) + noise) * temp)
        else:  # testing
            w = self.weight * self.subnet
            x = F.linear(x, w, self.bias)
        return x

import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import math


class VRPGE_Linear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scores = nn.Parameter(torch.Tensor(self.weight.shape))
        self.register_buffer('subnet', torch.zeros_like(self.scores))
        self.train_weights = False
        # nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))
        score_init_constant = 0.5
        self.scores.data = (
                torch.ones_like(self.scores) * score_init_constant
        )
        self.prune = True
        self.register_buffer("stored_mask_0", torch.zeros_like(self.scores))
        self.register_buffer("stored_mask_1", torch.zeros_like(self.scores))
        self.j = 0

    @property
    def clamped_scores(self):
        return self.scores

    def fix_subnet(self):
        self.subnet = (torch.rand_like(self.scores) < self.clamped_scores).float()

    def forward(self, x):
        # print(f'self.j: {self.j}')
        if self.prune:
            if not self.train_weights:
                self.subnet = StraightThroughBinomialSampleNoGrad.apply(self.scores) 
                if self.j == 0:
                    self.stored_mask_0.data = (self.subnet-self.scores)/torch.sqrt((self.scores+1e-20)*(1-self.scores+1e-20))
                else:
                    self.stored_mask_1.data = (self.subnet-self.scores)/torch.sqrt((self.scores+1e-20)*(1-self.scores+1e-20))
                w = self.weight * self.subnet
                print(f'w:{w}')
                print(f'bias:{self.bias}')
                x = F.linear(x, w, self.bias)
            else:
                w = self.weight * self.subnet
                x = F.linear(x, w, self.bias)
        else:
            x = F.linear(x, self.weight, self.bias)
        return x

class StraightThroughBinomialSampleNoGrad(autograd.Function):
    @staticmethod
    def forward(ctx, scores):
        output = (torch.rand_like(scores) < scores).float()
        return output

    @staticmethod
    def backward(ctx, grad_outputs):
        return torch.zeros_like(grad_outputs)