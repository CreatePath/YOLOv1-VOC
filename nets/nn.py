import sys
import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

from functools import partial
from torchvision.models import SwinTransformer

from config.net_config import SWIN_CONFIG, NET_CONFIG
from config.swin_config import SwinTransformerVersion



resnet50_url = 'https://download.pytorch.org/models/resnet50-19c8e357.pth',


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class DetNet(nn.Module):
    # no expansion
    # dilation = 2
    # type B use 1x1 conv
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, block_type='A'):
        super(DetNet, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=2, bias=False, dilation=2)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes or block_type == 'B':
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.downsample(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.in_planes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # self.layer5 = self._make_layer(block, 512, layers[3], stride=2)
        self.layer5 = self._make_detnet_layer(in_channels=2048)
        # self.avgpool = nn.AvgPool2d(14) #fit 448 input size
        # self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.conv_end = nn.Conv2d(256, 30, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_end = nn.BatchNorm2d(30)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        #[3, 4, 6, 3]
        layers = [block(self.in_planes, planes, stride, downsample)]
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def _make_detnet_layer(self, in_channels):
        layers = [
            DetNet(in_planes=in_channels, planes=256, block_type='B'),
            DetNet(in_planes=256, planes=256, block_type='A'),
            DetNet(in_planes=256, planes=256, block_type='A')
        ]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)
        x = self.conv_end(x)
        x = self.bn_end(x)
        x = torch.sigmoid(x)
        # x = x.view(-1,14,14,30)
        x = x.permute(0, 2, 3, 1)  # (-1,14,14,30)

        return x
    


class SwinTransformerHead(nn.Module):
    def __init__(self, inchannel: int, outchannel: int, outheight: int, outwidth: int):
        super(SwinTransformerHead, self).__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-5)

        self.conv1 = nn.Conv2d(inchannel, outchannel, 3, 1, 1)
        self.norm1 = norm_layer((outchannel, outheight, outwidth))

        self.conv2 = nn.Conv2d(outchannel, outchannel, 3, 1, 1)
        self.norm2 = norm_layer((outchannel, outheight, outwidth))

        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.gelu(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.gelu(x)

        return x



class SwinTransformerBasedYOLOv1(nn.Module):
    def __init__(self, cfg:dict, version:SwinTransformerVersion=None):
        super(SwinTransformerBasedYOLOv1, self).__init__()
        if version:
            backbone_cfg = cfg["BACKBONE"]["SWIN"][version]
            self.backbone = backbone_cfg["MODEL"]
            swin_outdim = backbone_cfg["EMBED_DIM"] * 8
        else:
            swin_cfg = cfg["BACKBONE"]["SWIN"]["CUSTOM"]
            swin_outdim = swin_cfg["EMBED_DIM"] * 8
            self.backbone = SwinTransformer(swin_cfg["PATCH_SIZE"],
                                            swin_cfg["EMBED_DIM"],
                                            swin_cfg["DEPTHS"],
                                            swin_cfg["NUM_HEADS"],
                                            swin_cfg["WINDOW_SIZE"],
                                            stochastic_depth_prob=swin_cfg["STOCHASTIC_DEPTH_PROB"],)

        self.backbone.avgpool = nn.AdaptiveAvgPool2d(cfg["OUTHEIGHT"])
        self.backbone.flatten = nn.Identity()
        self.backbone.head = SwinTransformerHead(swin_outdim,
                                                 cfg["OUTCHANNEL"],
                                                 cfg["OUTHEIGHT"],
                                                 cfg["OUTWIDTH"])
    
    def forward(self, x):
        x = self.backbone(x)
        return x.permute(0, 2, 3, 1)

# resnet50
def resnet50(pretrained=False, **kwargs):
    model_ = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model_.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/resnet50-19c8e357.pth'))
    return model_

# SwinTransformer
def swintransformer(cfg, version=None):
    model = SwinTransformerBasedYOLOv1(cfg, version)
    return model

if __name__ == '__main__':
    # a = torch.randn((2, 3, 448, 448))
    # model = resnet50()
    model = SwinTransformerBasedYOLOv1(NET_CONFIG, SWIN_CONFIG)
    print(model)
    # print(model(a).shape)
