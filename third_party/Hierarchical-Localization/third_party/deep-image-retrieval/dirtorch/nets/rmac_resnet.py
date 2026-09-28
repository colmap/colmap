import pdb
import torch
from .backbones.resnet import *
from .layers.pooling import GeneralizedMeanPooling, GeneralizedMeanPoolingP


def l2_normalize(x, axis=-1):
    x = F.normalize(x, p=2, dim=axis)
    return x


class ResNet_RMAC(ResNet):
    """ ResNet for RMAC (without ROI pooling)
    """
    def __init__(self, block, layers, model_name, out_dim=2048, norm_features=False,
                       pooling='gem', gemp=3, center_bias=0,
                       dropout_p=None, without_fc=False, **kwargs):
        ResNet.__init__(self, block, layers, 0, model_name, **kwargs)
        self.norm_features = norm_features
        self.without_fc = without_fc
        self.pooling = pooling
        self.center_bias = center_bias

        if pooling == 'max':
            self.adpool = nn.AdaptiveMaxPool2d(output_size=1)
        elif pooling == 'avg':
            self.adpool = nn.AdaptiveAvgPool2d(output_size=1)
        elif pooling.startswith('gem'):
            self.adpool = GeneralizedMeanPoolingP(norm=gemp)
        else:
            raise ValueError(pooling)

        self.dropout = nn.Dropout(dropout_p) if dropout_p is not None else None
        self.fc = nn.Linear(512 * block.expansion, out_dim)
        self.fc_name = 'fc'
        self.feat_dim = out_dim
        self.detach = False

    def forward(self, x):
        bs, _, H, W = x.shape

        x = ResNet.forward(self, x)

        if self.dropout is not None:
            x = self.dropout(x)

        if self.detach:
            # stop the back-propagation here, if needed
            x = Variable(x.detach())
            x = self.id(x)  # fake transformation

        if self.center_bias > 0:
            b = self.center_bias
            bias = 1 + torch.FloatTensor([[[[0,0,0,0],[0,b,b,0],[0,b,b,0],[0,0,0,0]]]]).to(x.device)
            bias = torch.nn.functional.interpolate(bias, size=x.shape[-2:], mode='bilinear', align_corners=True)
            x = x*bias

        # global pooling
        x = self.adpool(x)

        if self.norm_features:
            x = l2_normalize(x, axis=1)

        x.squeeze_()
        if not self.without_fc:
            x = self.fc(x)

        x = l2_normalize(x, axis=-1)
        return x




def resnet18_rmac(backbone=ResNet_RMAC, **kwargs):
    kwargs.pop('scales', None)
    return backbone(BasicBlock, [2, 2, 2, 2], 'resnet18', **kwargs)

def resnet50_rmac(backbone=ResNet_RMAC, **kwargs):
    kwargs.pop('scales', None)
    return backbone(Bottleneck, [3, 4, 6, 3], 'resnet50', **kwargs)

def resnet101_rmac(backbone=ResNet_RMAC, **kwargs):
    kwargs.pop('scales', None)
    return backbone(Bottleneck, [3, 4, 23, 3], 'resnet101', **kwargs)

def resnet152_rmac(backbone=ResNet_RMAC, **kwargs):
    kwargs.pop('scales', None)
    return backbone(Bottleneck, [3, 8, 36, 3], 'resnet152', **kwargs)





























