# Copyright 2019-present NAVER Corp.
# CC BY-NC-SA 3.0
# Available only for non-commercial use

import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F

from nets.sampler import *
from nets.repeatability_loss import *
from nets.reliability_loss import *


class MultiLoss (nn.Module):
    """ Combines several loss functions for convenience.
    *args: [loss weight (float), loss creator, ... ]
    
    Example:
        loss = MultiLoss( 1, MyFirstLoss(), 0.5, MySecondLoss() )
    """
    def __init__(self, *args, dbg=()):
        nn.Module.__init__(self)
        assert len(args) % 2 == 0, 'args must be a list of (float, loss)'
        self.weights = []
        self.losses = nn.ModuleList()
        for i in range(len(args)//2):
            weight = float(args[2*i+0])
            loss = args[2*i+1]
            assert isinstance(loss, nn.Module), "%s is not a loss!" % loss
            self.weights.append(weight)
            self.losses.append(loss)

    def forward(self, select=None, **variables):
        assert not select or all(1<=n<=len(self.losses) for n in select)
        d = dict()
        cum_loss = 0
        for num, (weight, loss_func) in enumerate(zip(self.weights, self.losses),1):
            if select is not None and num not in select: continue
            l = loss_func(**{k:v for k,v in variables.items()})
            if isinstance(l, tuple):
                assert len(l) == 2 and isinstance(l[1], dict)
            else:
                l = l, {loss_func.name:l}
            cum_loss = cum_loss + weight * l[0]
            for key,val in l[1].items():
                d['loss_'+key] = float(val)
        d['loss'] = float(cum_loss)
        return cum_loss, d






