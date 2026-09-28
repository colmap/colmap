import pdb
from .backbones.resnet import *
from .layers.pooling import GeneralizedMeanPooling, GeneralizedMeanPoolingP


def l2_normalize(x, axis=-1):
    x = F.normalize(x, p=2, dim=axis)
    return x


class ResNet_RMAC_FPN(ResNet):
    """ ResNet for RMAC (without ROI pooling)
    """
    def __init__(self, block, layers, model_name, out_dim=None, norm_features=False,
                       pooling='gem', gemp=3, center_bias=0, mode=1,
                       dropout_p=None, without_fc=False, **kwargs):
        ResNet.__init__(self, block, layers, 0, model_name, **kwargs)
        self.norm_features = norm_features
        self.without_fc = without_fc
        self.pooling = pooling
        self.center_bias = center_bias
        self.mode = mode

        dim1 = 256 * block.expansion
        dim2 = 512 * block.expansion
        if out_dim is None: out_dim = dim1 + dim2
        #FPN
        if self.mode == 1:
            self.conv1x5 = nn.Conv2d(dim2, dim1, kernel_size=1, stride=1, bias=False)
            self.conv3c4 = nn.Conv2d(dim1, dim1, kernel_size=3, stride=1, padding=1, bias=False)
            self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

        if pooling == 'max':
            self.adpool = nn.AdaptiveMaxPool2d(output_size=1)
        elif pooling == 'avg':
            self.adpool = nn.AdaptiveAvgPool2d(output_size=1)
        elif pooling == 'gem':
            self.adpoolx5 = GeneralizedMeanPoolingP(norm=gemp)
            self.adpoolc4 = GeneralizedMeanPoolingP(norm=gemp)

        self.dropout = nn.Dropout(dropout_p) if dropout_p is not None else None
        self.fc = nn.Linear(768 * block.expansion, out_dim)
        self.fc_name = 'fc'
        self.feat_dim = out_dim
        self.detach = False

    def forward(self, x):
        x4, x5 = ResNet.forward(self, x, -1)

        # FPN
        if self.mode == 1:
            c5 = F.interpolate(x5, size=x4.shape[-2:], mode='nearest')

            c5 = self.conv1x5(c5)
            c5 = self.relu(c5)
            x4 = x4 + c5
            x4 = self.conv3c4(x4)
            x4 = self.relu(x4)

        if self.dropout is not None:
            x5 = self.dropout(x5)
            x4 = self.dropout(x4)

        if self.detach:
            # stop the back-propagation here, if needed
            x5 = Variable(x5.detach())
            x5 = self.id(x5)  # fake transformation
            x4 = Variable(x4.detach())
            x4 = self.id(x4)  # fake transformation

        # global pooling
        x5 = self.adpoolx5(x5)
        x4 = self.adpoolc4(x4)

        x = torch.cat((x4, x5), 1)

        if self.norm_features:
            x = l2_normalize(x, axis=1)

        x.squeeze_()
        if not self.without_fc:
            x = self.fc(x)

        x = l2_normalize(x, axis=-1)
        return x




def resnet18_fpn_rmac(backbone=ResNet_RMAC_FPN, **kwargs):
    kwargs.pop('scales', None)
    return backbone(BasicBlock, [2, 2, 2, 2], 'resnet18', **kwargs)

def resnet50_fpn_rmac(backbone=ResNet_RMAC_FPN, **kwargs):
    kwargs.pop('scales', None)
    return backbone(Bottleneck, [3, 4, 6, 3], 'resnet50', **kwargs)

def resnet101_fpn_rmac(backbone=ResNet_RMAC_FPN, **kwargs):
    kwargs.pop('scales', None)
    return backbone(Bottleneck, [3, 4, 23, 3], 'resnet101', **kwargs)

def resnet101_fpn0_rmac(backbone=ResNet_RMAC_FPN, **kwargs):
    kwargs.pop('scales', None)
    return backbone(Bottleneck, [3, 4, 23, 3], 'resnet101', mode=0, **kwargs)

def resnet152_fpn_rmac(backbone=ResNet_RMAC_FPN, **kwargs):
    kwargs.pop('scales', None)
    return backbone(Bottleneck, [3, 8, 36, 3], 'resnet152', **kwargs)





























