# Copyright 2019-present NAVER Corp.
# CC BY-NC-SA 3.0
# Available only for non-commercial use

import pdb

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


""" Different samplers, each specifying how to sample pixels for the AP loss.
"""


class FullSampler(nn.Module):
    """ all pixels are selected
        - feats: keypoint descriptors
        - confs: reliability values
    """
    def __init__(self):
        nn.Module.__init__(self)
        self.mode = 'bilinear'
        self.padding = 'zeros'

    @staticmethod
    def _aflow_to_grid(aflow):
        H, W = aflow.shape[2:]
        grid = aflow.permute(0,2,3,1).clone()
        grid[:,:,:,0] *= 2/(W-1)
        grid[:,:,:,1] *= 2/(H-1)
        grid -= 1
        grid[torch.isnan(grid)] = 9e9 # invalids
        return grid
    
    def _warp(self, feats, confs, aflow):
        if isinstance(aflow, tuple): return aflow # result was precomputed
        feat1, feat2 = feats
        conf1, conf2 = confs if confs else (None,None)
    
        B, two, H, W = aflow.shape
        D = feat1.shape[1]
        assert feat1.shape == feat2.shape == (B, D, H, W) # D = 128, B = batch
        assert conf1.shape == conf2.shape == (B, 1, H, W) if confs else True

        # warp img2 to img1
        grid = self._aflow_to_grid(aflow)
        ones2 = feat2.new_ones(feat2[:,0:1].shape)
        feat2to1 = F.grid_sample(feat2, grid, mode=self.mode, padding_mode=self.padding)
        mask2to1 = F.grid_sample(ones2, grid, mode='nearest', padding_mode='zeros')
        conf2to1 = F.grid_sample(conf2, grid, mode=self.mode, padding_mode=self.padding) \
                   if confs else None
        return feat2to1, mask2to1.byte(), conf2to1

    def _warp_positions(self, aflow):
        B, two, H, W = aflow.shape
        assert two == 2
        
        Y = torch.arange(H, device=aflow.device)
        X = torch.arange(W, device=aflow.device)
        XY = torch.stack(torch.meshgrid(Y,X)[::-1], dim=0)
        XY = XY[None].expand(B, 2, H, W).float()
        
        grid = self._aflow_to_grid(aflow)
        XY2 = F.grid_sample(XY, grid, mode='bilinear', padding_mode='zeros')
        return XY, XY2



class SubSampler (FullSampler):
    """ pixels are selected in an uniformly spaced grid
    """
    def __init__(self, border, subq, subd, perimage=False):
        FullSampler.__init__(self)
        assert subq % subd == 0, 'subq must be multiple of subd'
        self.sub_q = subq
        self.sub_d = subd
        self.border = border
        self.perimage = perimage

    def __repr__(self):
        return "SubSampler(border=%d, subq=%d, subd=%d, perimage=%d)" % (
            self.border, self.sub_q, self.sub_d, self.perimage)

    def __call__(self, feats, confs, aflow):
        feat1, conf1 = feats[0], (confs[0] if confs else None)
        # warp with optical flow in img1 coords
        feat2, mask2, conf2 = self._warp(feats, confs, aflow)
        
        # subsample img1
        slq = slice(self.border, -self.border or None, self.sub_q)
        feat1 = feat1[:, :, slq, slq]
        conf1 = conf1[:, :, slq, slq] if confs else None
        # subsample img2
        sld = slice(self.border, -self.border or None, self.sub_d)
        feat2 = feat2[:, :, sld, sld]
        mask2 = mask2[:, :, sld, sld]
        conf2 = conf2[:, :, sld, sld] if confs else None
        
        B, D, Hq, Wq = feat1.shape
        B, D, Hd, Wd = feat2.shape
        
        # compute gt
        if self.perimage or self.sub_q != self.sub_d:
            # compute ground-truth by comparing pixel indices
            f = feats[0][0:1,0] if self.perimage else feats[0][:,0]
            idxs = torch.arange(f.numel(), dtype=torch.int64, device=feat1.device).view(f.shape)
            idxs1 = idxs[:, slq, slq].reshape(-1,Hq*Wq)
            idxs2 = idxs[:, sld, sld].reshape(-1,Hd*Wd)
            if self.perimage:
                gt = (idxs1[0].view(-1,1) == idxs2[0].view(1,-1))
                gt = gt[None,:,:].expand(B, Hq*Wq, Hd*Wd)
            else :
                gt = (idxs1.view(-1,1) == idxs2.view(1,-1)) 
        else:
            gt = torch.eye(feat1[:,0].numel(), dtype=torch.uint8, device=feat1.device) # always binary for AP loss
        
        # compute all images together
        queries  =  feat1.reshape(B,D,-1) # B x D x (Hq x Wq)
        database =  feat2.reshape(B,D,-1) # B x D x (Hd x Wd)
        if self.perimage:
            queries  =  queries.transpose(1,2) # B x (Hd x Wd) x D
            scores = torch.bmm(queries, database) # B x (Hq x Wq) x (Hd x Wd)
        else:
            queries  =  queries .transpose(1,2).reshape(-1,D) # (B x Hq x Wq) x D
            database =  database.transpose(1,0).reshape(D,-1) # D x (B x Hd x Wd)
            scores = torch.matmul(queries, database) # (B x Hq x Wq) x (B x Hd x Wd)

        # compute reliability
        qconf = (conf1 + conf2)/2 if confs else None

        assert gt.shape == scores.shape
        return scores, gt, mask2, qconf



class NghSampler (FullSampler):
    """ all pixels in a small neighborhood
    """
    def __init__(self, ngh, subq=1, subd=1, ignore=1, border=None):
        FullSampler.__init__(self)
        assert 0 <= ignore < ngh
        self.ngh = ngh
        self.ignore = ignore
        assert subd <= ngh
        self.sub_q = subq
        self.sub_d = subd
        if border is None: border = ngh
        assert border >= ngh, 'border has to be larger than ngh'
        self.border = border

    def __repr__(self):
        return "NghSampler(ngh=%d, subq=%d, subd=%d, ignore=%d, border=%d)" % (
            self.ngh, self.sub_q, self.sub_d, self.ignore, self.border)

    def trans(self, arr, i, j):
        s = lambda i: slice(self.border+i, i-self.border or None, self.sub_q)
        return arr[:,:,s(j),s(i)]

    def __call__(self, feats, confs, aflow):
        feat1, conf1 = feats[0], (confs[0] if confs else None)
        # warp with optical flow in img1 coords
        feat2, mask2, conf2 = self._warp(feats, confs, aflow)
        
        qfeat = self.trans(feat1,0,0)
        qconf = (self.trans(conf1,0,0) + self.trans(conf2,0,0)) / 2 if confs else None
        mask2 = self.trans(mask2,0,0)
        scores_at = lambda i,j: (qfeat * self.trans(feat2,i,j)).sum(dim=1)
        
        # compute scores for all neighbors
        B, D = feat1.shape[:2]
        min_d = self.ignore**2
        max_d = self.ngh**2
        rad = (self.ngh//self.sub_d) * self.ngh # make an integer multiple
        negs = []
        offsets = []
        for j in range(-rad, rad+1, self.sub_d):
          for i in range(-rad, rad+1, self.sub_d):
            if not(min_d < i*i + j*j <= max_d): 
                continue # out of scope
            offsets.append((i,j)) # Note: this list is just for debug
            negs.append( scores_at(i,j) )
        
        scores = torch.stack([scores_at(0,0)] + negs, dim=-1)
        gt = scores.new_zeros(scores.shape, dtype=torch.uint8)
        gt[..., 0] = 1 # only the center point is positive

        return scores, gt, mask2, qconf



class FarNearSampler (FullSampler):
    """ Sample pixels from *both* a small neighborhood *and* far-away pixels.
        
    How it works?
        1) Queries are sampled from img1,
            - at least `border` pixels from borders and 
            - on a grid with step = `subq`
            
        2) Close database pixels 
            - from the corresponding image (img2),
            - within a `ngh` distance radius 
            - on a grid with step = `subd_ngh`
            - ignored if distance to query is >0 and <=`ignore`
            
        3) Far-away database pixels from ,
            - from all batch images in `img2`
            - at least `border` pixels from borders
            - on a grid with step = `subd_far`
    """
    def __init__(self, subq, ngh, subd_ngh, subd_far, border=None, ignore=1, 
                       maxpool_ngh=False ):
        FullSampler.__init__(self)
        border = border or ngh
        assert ignore < ngh < subd_far, 'neighborhood needs to be smaller than far step'
        self.close_sampler = NghSampler(ngh=ngh, subq=subq, subd=subd_ngh, 
                ignore=not(maxpool_ngh), border=border)
        self.faraway_sampler = SubSampler(border=border, subq=subq, subd=subd_far)
        self.maxpool_ngh = maxpool_ngh

    def __repr__(self):
        c,f = self.close_sampler, self.faraway_sampler
        res = "FarNearSampler(subq=%d, ngh=%d" % (c.sub_q, c.ngh)
        res += ", subd_ngh=%d, subd_far=%d" % (c.sub_d, f.sub_d)
        res += ", border=%d, ign=%d" % (f.border, c.ignore)
        res += ", maxpool_ngh=%d" % self.maxpool_ngh
        return res+')'

    def __call__(self, feats, confs, aflow):
        # warp with optical flow in img1 coords
        aflow = self._warp(feats, confs, aflow)

        # sample ngh pixels
        scores1, gt1, msk1, conf1 = self.close_sampler(feats, confs, aflow)
        scores1, gt1 = scores1.view(-1,scores1.shape[-1]), gt1.view(-1,gt1.shape[-1])
        if self.maxpool_ngh:
            # we consider all scores from ngh as potential positives
            scores1, self._cached_maxpool_ngh = scores1.max(dim=1,keepdim=True)
            gt1 = gt1[:, 0:1]

        # sample far pixels
        scores2, gt2, msk2, conf2 = self.faraway_sampler(feats, confs, aflow)
        # assert (msk1 == msk2).all()
        # assert (conf1 == conf2).all()

        return (torch.cat((scores1,scores2),dim=1), 
                torch.cat((gt1,    gt2),    dim=1), 
                msk1, conf1 if confs else None)


class NghSampler2 (nn.Module):
    """ Similar to NghSampler, but doesnt warp the 2nd image.
    Distance to GT =>  0 ... pos_d ... neg_d ... ngh
    Pixel label    =>  + + + + + + 0 0 - - - - - - -
    
    Subsample on query side: if > 0, regular grid
                                < 0, random points 
    In both cases, the number of query points is = W*H/subq**2
    """
    def __init__(self, ngh, subq=1, subd=1, pos_d=0, neg_d=2, border=None,
                       maxpool_pos=True, subd_neg=0):
        nn.Module.__init__(self)
        assert 0 <= pos_d < neg_d <= (ngh if ngh else 99)
        self.ngh = ngh
        self.pos_d = pos_d
        self.neg_d = neg_d
        assert subd <= ngh or ngh == 0
        assert subq != 0
        self.sub_q = subq
        self.sub_d = subd
        self.sub_d_neg = subd_neg
        if border is None: border = ngh
        assert border >= ngh, 'border has to be larger than ngh'
        self.border = border
        self.maxpool_pos = maxpool_pos
        self.precompute_offsets()

    def precompute_offsets(self):
        pos_d2 = self.pos_d**2
        neg_d2 = self.neg_d**2
        rad2 = self.ngh**2
        rad = (self.ngh//self.sub_d) * self.ngh # make an integer multiple
        pos = []
        neg = []
        for j in range(-rad, rad+1, self.sub_d):
          for i in range(-rad, rad+1, self.sub_d):
            d2 = i*i + j*j
            if d2 <= pos_d2:
                pos.append( (i,j) )
            elif neg_d2 <= d2 <= rad2: 
                neg.append( (i,j) )

        self.register_buffer('pos_offsets', torch.LongTensor(pos).view(-1,2).t())
        self.register_buffer('neg_offsets', torch.LongTensor(neg).view(-1,2).t())

    def gen_grid(self, step, aflow):
        B, two, H, W = aflow.shape
        dev = aflow.device
        b1 = torch.arange(B, device=dev)
        if step > 0:
            # regular grid
            x1 = torch.arange(self.border, W-self.border, step, device=dev)
            y1 = torch.arange(self.border, H-self.border, step, device=dev)
            H1, W1 = len(y1), len(x1)
            x1 = x1[None,None,:].expand(B,H1,W1).reshape(-1)
            y1 = y1[None,:,None].expand(B,H1,W1).reshape(-1)
            b1 = b1[:,None,None].expand(B,H1,W1).reshape(-1)
            shape = (B, H1, W1)
        else:
            # randomly spread
            n = (H - 2*self.border) * (W - 2*self.border) // step**2
            x1 = torch.randint(self.border, W-self.border, (n,), device=dev)
            y1 = torch.randint(self.border, H-self.border, (n,), device=dev)
            x1 = x1[None,:].expand(B,n).reshape(-1)
            y1 = y1[None,:].expand(B,n).reshape(-1)
            b1 = b1[:,None].expand(B,n).reshape(-1)
            shape = (B, n)
        return b1, y1, x1, shape

    def forward(self, feats, confs, aflow, **kw):
        B, two, H, W = aflow.shape
        assert two == 2
        feat1, conf1 = feats[0], (confs[0] if confs else None)
        feat2, conf2 = feats[1], (confs[1] if confs else None)
        
        # positions in the first image
        b1, y1, x1, shape = self.gen_grid(self.sub_q, aflow)

        # sample features from first image
        feat1 = feat1[b1, :, y1, x1]
        qconf = conf1[b1, :, y1, x1].view(shape) if confs else None
        
        #sample GT from second image
        b2 = b1
        xy2 = (aflow[b1, :, y1, x1] + 0.5).long().t()
        mask = (0 <= xy2[0]) * (0 <= xy2[1]) * (xy2[0] < W) * (xy2[1] < H)
        mask = mask.view(shape)
        
        def clamp(xy):
            torch.clamp(xy[0], 0, W-1, out=xy[0])
            torch.clamp(xy[1], 0, H-1, out=xy[1])
            return xy
        
        # compute positive scores
        xy2p = clamp(xy2[:,None,:] + self.pos_offsets[:,:,None])
        pscores = (feat1[None,:,:] * feat2[b2, :, xy2p[1], xy2p[0]]).sum(dim=-1).t()
#        xy1p = clamp(torch.stack((x1,y1))[:,None,:] + self.pos_offsets[:,:,None])
#        grid = FullSampler._aflow_to_grid(aflow)
#        feat2p = F.grid_sample(feat2, grid, mode='bilinear', padding_mode='border')
#        pscores = (feat1[None,:,:] * feat2p[b1,:,xy1p[1], xy1p[0]]).sum(dim=-1).t()
        if self.maxpool_pos:
            pscores, pos = pscores.max(dim=1, keepdim=True)
            if confs: 
                sel = clamp(xy2 + self.pos_offsets[:,pos.view(-1)])
                qconf = (qconf + conf2[b2, :, sel[1], sel[0]].view(shape))/2
        
        # compute negative scores
        xy2n = clamp(xy2[:,None,:] + self.neg_offsets[:,:,None])
        nscores = (feat1[None,:,:] * feat2[b2, :, xy2n[1], xy2n[0]]).sum(dim=-1).t()

        if self.sub_d_neg:
            # add distractors from a grid
            b3, y3, x3, _ = self.gen_grid(self.sub_d_neg, aflow)
            distractors = feat2[b3, :, y3, x3]
            dscores = torch.matmul(feat1, distractors.t())
            del distractors
            
            # remove scores that corresponds to positives or nulls
            dis2 = (x3 - xy2[0][:,None])**2 + (y3 - xy2[1][:,None])**2
            dis2 += (b3 != b2[:,None]).long() * self.neg_d**2
            dscores[dis2 < self.neg_d**2] = 0
            
            scores = torch.cat((pscores, nscores, dscores), dim=1)
        else:
            # concat everything
            scores = torch.cat((pscores, nscores), dim=1)

        gt = scores.new_zeros(scores.shape, dtype=torch.uint8)
        gt[:, :pscores.shape[1]] = 1

        return scores, gt, mask, qconf








