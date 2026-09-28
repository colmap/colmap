from .backbones.resnext101_features import *
from .layers.pooling import GeneralizedMeanPooling, GeneralizedMeanPoolingP


def l2_normalize(x, axis=-1):
    x = F.normalize(x, p=2, dim=axis)
    return x


class ResNext_RMAC(nn.Module):
    """ ResNet for RMAC (without ROI pooling)
    """
    def __init__(self, backbone, out_dim=2048, norm_features=False,
                       pooling='gem', gemp=3, center_bias=0,
                       dropout_p=None, without_fc=False, **kwargs):
        super(ResNeXt_RMAC, self).__init__()
        self.backbone = backbone
        self.norm_features = norm_features
        self.without_fc = without_fc
        self.pooling = pooling
        self.center_bias = center_bias

        if pooling == 'max':
            self.adpool = nn.AdaptiveMaxPool2d(output_size=1)
        elif pooling == 'avg':
            self.adpool = nn.AdaptiveAvgPool2d(output_size=1)
        elif pooling == 'gem':
            self.adpool = GeneralizedMeanPoolingP(norm=gemp)

        self.dropout = nn.Dropout(dropout_p) if dropout_p is not None else None
        self.last_linear = nn.Linear(2048, out_dim)
        self.fc_name = 'last_linear'
        self.feat_dim = out_dim
        self.detach = False

    def forward(self, x):
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





























