""" Load required packages """

import torch
import numpy as np
import scipy.io
import h5py
import torch.nn as nn

import operator
from functools import reduce
from functools import partial


""" Utility functions for --loading data, 
                          --data normalization, 
                          --data standerdization, and 
                          --loss evaluation 

The base codes are taken from the repo: https://github.com/zongyi-li/fourier_neural_operator
"""

device = torch.device('cuda:1')


# normalization, pointwise gaussian
class UnitGaussianNormalizer:
    def __init__(self, tensor):
        # channel-wise normalization (BEST for FNO)
        self.mean = tensor.mean(dim=(0,1,2), keepdim=True)
        self.std  = tensor.std(dim=(0,1,2), keepdim=True)
        self.eps = 1e-5

    def encode(self, tensor):
        return (tensor - self.mean) / (self.std + self.eps)

    def decode(self, tensor):
        return tensor * (self.std + self.eps) + self.mean

    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)

# normalization, scaling by range
class RangeNormalizer(object):
    def __init__(self, x, low=0.0, high=1.0):
        super(RangeNormalizer, self).__init__()
        mymin = torch.min(x, 0)[0].view(-1)
        mymax = torch.max(x, 0)[0].view(-1)

        self.a = (high - low)/(mymax - mymin)
        self.b = -self.a*mymax + high

    def encode(self, x):
        s = x.size()
        x = x.view(s[0], -1)
        x = self.a*x + self.b
        x = x.view(s)
        return x

    def decode(self, x):
        s = x.size()
        x = x.view(s[0], -1)
        x = (x - self.b)/self.a
        x = x.view(s)
        return x

# loss function with rel/abs Lp loss
class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        #Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h**(self.d/self.p))*torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)

        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.rel(x, y)
    
class LogNormalizer(object):
    def __init__(self, eps=1e-8):
        self.mean = None
        self.std = None
        self.eps = eps

    def fit(self, y):
        """
        y: torch tensor (N, H, W)
        """
        y_log = torch.log1p(y)
        self.mean = y_log.mean()
        self.std = y_log.std() + self.eps

    def encode(self, y):
        y_log = torch.log1p(y)
        return (y_log - self.mean) / self.std

    def decode(self, y_norm):
        y_log = y_norm * self.std + self.mean
        return torch.expm1(y_log)

    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)

def count_params(model, trainable_only=True, count_complex_as_two=True):
    total = 0
    for p in model.parameters():
        if trainable_only and not p.requires_grad:
            continue
        n = p.numel()
        if count_complex_as_two and p.is_complex():
            n *= 2
        total += n
    return total


