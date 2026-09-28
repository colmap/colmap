# Copyright 2019-present NAVER Corp.
# CC BY-NC-SA 3.0
# Available only for non-commercial use

import pdb
import torch.nn as nn
import torch.nn.functional as F

from nets.ap_loss import APLoss


class PixelAPLoss (nn.Module):
    """ Computes the pixel-wise AP loss:
        Given two images and ground-truth optical flow, computes the AP per pixel.
        
        feat1:  (B, C, H, W)   pixel-wise features extracted from img1
        feat2:  (B, C, H, W)   pixel-wise features extracted from img2
        aflow:  (B, 2, H, W)   absolute flow: aflow[...,y1,x1] = x2,y2
    """
    def __init__(self, sampler, nq=20):
        nn.Module.__init__(self)
        self.aploss = APLoss(nq, min=0, max=1, euc=False)
        self.name = 'pixAP'
        self.sampler = sampler

    def loss_from_ap(self, ap, rel):
        return 1 - ap

    def forward(self, descriptors, aflow, **kw):
        # subsample things
        scores, gt, msk, qconf = self.sampler(descriptors, kw.get('reliability'), aflow)
        
        # compute pixel-wise AP
        n = qconf.numel()
        if n == 0: return 0
        scores, gt = scores.view(n,-1), gt.view(n,-1)
        ap = self.aploss(scores, gt).view(msk.shape)

        pixel_loss = self.loss_from_ap(ap, qconf)
        
        loss = pixel_loss[msk].mean()
        return loss


class ReliabilityLoss (PixelAPLoss):
    """ same than PixelAPLoss, but also train a pixel-wise confidence
        that this pixel is going to have a good AP.
    """
    def __init__(self, sampler, base=0.5, **kw):
        PixelAPLoss.__init__(self, sampler, **kw)
        assert 0 <= base < 1
        self.base = base
        self.name = 'reliability'

    def loss_from_ap(self, ap, rel):
        return 1 - ap*rel - (1-rel)*self.base



