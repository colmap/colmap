import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class APLoss (nn.Module):
    """ Differentiable AP loss, through quantization. From the paper:

        Learning with Average Precision: Training Image Retrieval with a Listwise Loss
        Jerome Revaud, Jon Almazan, Rafael Sampaio de Rezende, Cesar de Souza
        https://arxiv.org/abs/1906.07589

        Input: (N, M)   values in [min, max]
        label: (N, M)   values in {0, 1}

        Returns: 1 - mAP (mean AP for each n in {1..N})
                 Note: typically, this is what you wanna minimize
    """
    def __init__(self, nq=25, min=0, max=1):
        nn.Module.__init__(self)
        assert isinstance(nq, int) and 2 <= nq <= 100
        self.nq = nq
        self.min = min
        self.max = max
        gap = max - min
        assert gap > 0
        # Initialize quantizer as non-trainable convolution
        self.quantizer = q = nn.Conv1d(1, 2*nq, kernel_size=1, bias=True)
        q.weight = nn.Parameter(q.weight.detach(), requires_grad=False)
        q.bias = nn.Parameter(q.bias.detach(), requires_grad=False)
        a = (nq-1) / gap
        # First half equal to lines passing to (min+x,1) and (min+x+1/a,0) with x = {nq-1..0}*gap/(nq-1)
        q.weight[:nq] = -a
        q.bias[:nq] = torch.from_numpy(a*min + np.arange(nq, 0, -1))  # b = 1 + a*(min+x)
        # First half equal to lines passing to (min+x,1) and (min+x-1/a,0) with x = {nq-1..0}*gap/(nq-1)
        q.weight[nq:] = a
        q.bias[nq:] = torch.from_numpy(np.arange(2-nq, 2, 1) - a*min)  # b = 1 - a*(min+x)
        # First and last one as a horizontal straight line
        q.weight[0] = q.weight[-1] = 0
        q.bias[0] = q.bias[-1] = 1

    def forward(self, x, label, qw=None, ret='1-mAP'):
        assert x.shape == label.shape  # N x M
        N, M = x.shape
        # Quantize all predictions
        q = self.quantizer(x.unsqueeze(1))
        q = torch.min(q[:, :self.nq], q[:, self.nq:]).clamp(min=0)  # N x Q x M

        nbs = q.sum(dim=-1)  # number of samples  N x Q = c
        rec = (q * label.view(N, 1, M).float()).sum(dim=-1)  # number of correct samples = c+ N x Q
        prec = rec.cumsum(dim=-1) / (1e-16 + nbs.cumsum(dim=-1))  # precision
        rec /= rec.sum(dim=-1).unsqueeze(1)  # norm in [0,1]

        ap = (prec * rec).sum(dim=-1)  # per-image AP

        if ret == '1-mAP':
            if qw is not None:
                ap *= qw  # query weights
            return 1 - ap.mean()
        elif ret == 'AP':
            assert qw is None
            return ap
        else:
            raise ValueError("Bad return type for APLoss(): %s" % str(ret))

    def measures(self, x, gt, loss=None):
        if loss is None:
            loss = self.forward(x, gt)
        return {'loss_ap': float(loss)}


class TAPLoss (APLoss):
    """ Differentiable tie-aware AP loss, through quantization. From the paper:

        Learning with Average Precision: Training Image Retrieval with a Listwise Loss
        Jerome Revaud, Jon Almazan, Rafael Sampaio de Rezende, Cesar de Souza
        https://arxiv.org/abs/1906.07589

        Input: (N, M)   values in [min, max]
        label: (N, M)   values in {0, 1}

        Returns: 1 - mAP (mean AP for each n in {1..N})
                 Note: typically, this is what you wanna minimize
    """
    def __init__(self, nq=25, min=0, max=1, simplified=False):
        APLoss.__init__(self, nq=nq, min=min, max=max)
        self.simplified = simplified

    def forward(self, x, label, qw=None, ret='1-mAP'):
        '''N: number of images;
           M: size of the descs;
           Q: number of bins (nq);
        '''
        assert x.shape == label.shape  # N x M
        N, M = x.shape
        label = label.float()
        Np = label.sum(dim=-1, keepdim=True)

        # Quantize all predictions
        q = self.quantizer(x.unsqueeze(1))
        q = torch.min(q[:, :self.nq], q[:, self.nq:]).clamp(min=0)  # N x Q x M

        c = q.sum(dim=-1)  # number of samples  N x Q = nbs on APLoss
        cp = (q * label.view(N, 1, M)).sum(dim=-1)  # N x Q number of correct samples = rec on APLoss
        C = c.cumsum(dim=-1)
        Cp = cp.cumsum(dim=-1)

        zeros = torch.zeros(N, 1).to(x.device)
        C_1d = torch.cat((zeros, C[:, :-1]), dim=-1)
        Cp_1d = torch.cat((zeros, Cp[:, :-1]), dim=-1)

        if self.simplified:
            aps = cp * (Cp_1d+Cp+1) / (C_1d+C+1) / Np
        else:
            eps = 1e-8
            ratio = (cp - 1).clamp(min=0) / ((c-1).clamp(min=0) + eps)
            aps = cp * (c * ratio + (Cp_1d + 1 - ratio * (C_1d + 1)) * torch.log((C + 1) / (C_1d + 1))) / (c + eps) / Np
        aps = aps.sum(dim=-1)

        assert aps.numel() == N

        if ret == '1-mAP':
            if qw is not None:
                aps *= qw  # query weights
            return 1 - aps.mean()
        elif ret == 'AP':
            assert qw is None
            return aps
        else:
            raise ValueError("Bad return type for APLoss(): %s" % str(ret))

    def measures(self, x, gt, loss=None):
        if loss is None:
            loss = self.forward(x, gt)
        return {'loss_tap'+('s' if self.simplified else ''): float(loss)}


class TripletMarginLoss(nn.TripletMarginLoss):
    """ PyTorch's margin triplet loss

    TripletMarginLoss(margin=1.0, p=2, eps=1e-06, swap=False, size_average=True, reduce=True)
    """
    def eval_func(self, dp, dn):
        return max(0, dp - dn + self.margin)


class TripletLogExpLoss(nn.Module):
    r"""Creates a criterion that measures the triplet loss given an input
    tensors x1, x2, x3.
    This is used for measuring a relative similarity between samples. A triplet
    is composed by `a`, `p` and `n`: anchor, positive examples and negative
    example respectively. The shape of all input variables should be
    :math:`(N, D)`.

    The distance is described in detail in the paper `Improving Pairwise Ranking for Multi-Label
    Image Classification`_ by Y. Li et al.

    .. math::
        L(a, p, n) = log \left( 1 + exp(d(a_i, p_i) - d(a_i, n_i) \right)


    Args:
        anchor: anchor input tensor
        positive: positive input tensor
        negative: negative input tensor

    Shape:
        - Input: :math:`(N, D)` where `D = vector dimension`
        - Output: :math:`(N, 1)`

    >>> triplet_loss = nn.TripletLogExpLoss(p=2)
    >>> input1 = autograd.Variable(torch.randn(100, 128))
    >>> input2 = autograd.Variable(torch.randn(100, 128))
    >>> input3 = autograd.Variable(torch.randn(100, 128))
    >>> output = triplet_loss(input1, input2, input3)
    >>> output.backward()

    .. _Learning shallow convolutional feature descriptors with triplet losses:
        http://www.iis.ee.ic.ac.uk/%7Evbalnt/shallow_descr/TFeat_paper.pdf
    """

    def __init__(self, p=2, eps=1e-6, swap=False):
        super(TripletLogExpLoss, self).__init__()
        self.p = p
        self.eps = eps
        self.swap = swap

    def forward(self, anchor, positive, negative):
        assert anchor.size() == positive.size(), "Input sizes between positive and negative must be equal."
        assert anchor.size() == negative.size(), "Input sizes between anchor and negative must be equal."
        assert positive.size() == negative.size(), "Input sizes between positive and negative must be equal."
        assert anchor.dim() == 2, "Input must be a 2D matrix."

        d_p = F.pairwise_distance(anchor, positive, self.p, self.eps)
        d_n = F.pairwise_distance(anchor, negative, self.p, self.eps)

        if self.swap:
            d_s = F.pairwise_distance(positive, negative, self.p, self.eps)
            d_n = torch.min(d_n, d_s)

        dist = torch.log(1 + torch.exp(d_p - d_n))
        loss = torch.mean(dist)
        return loss

    def eval_func(self, dp, dn):
        return np.log(1 + np.exp(dp - dn))


def sim_to_dist(scores):
    return 1 - torch.sqrt(2.001 - 2*scores)


class APLoss_dist (APLoss):
    def forward(self, x, label, **kw):
        d = sim_to_dist(x)
        return APLoss.forward(self, d, label, **kw)


class TAPLoss_dist (TAPLoss):
    def forward(self, x, label, **kw):
        d = sim_to_dist(x)
        return TAPLoss.forward(self, d, label, **kw)
