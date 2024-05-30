import math
import torch
from torch import nn
from timm.models.layers import DropPath

from functools import partial

from nets.nn import DetNet

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self,
                 inplanes:int,
                 planes:int,
                 stride:int=1,
                 groups:int=1,
                 droppath:float=0) -> None:
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, 7, stride, 3, groups=groups)
        self.conv2 = nn.Conv2d(planes, planes * self.expansion, 1)
        self.conv3 = nn.Conv2d(planes * self.expansion, planes, 1)

        self.norm = nn.InstanceNorm2d(inplanes)
        self.activation = nn.GELU()
        self.droppath = DropPath(droppath)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.norm(out)

        out = self.conv2(out)
        out = self.activation(out)

        out = self.conv3(out)
        out = residual + self.droppath(out)
        return out



class Conv1x1Decoder(nn.Module):
    def __init__(self, inplanes, planes) -> None:
        super(Conv1x1Decoder, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, 512, 1)
        self.conv2 = nn.Conv2d(512, 256, 1)
        self.conv3 = nn.Conv2d(256, 128, 1)
        self.conv4 = nn.Conv2d(128, planes, 1)
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        x = self.activation(x)

        x = self.conv3(x)
        x = self.conv4(x)
        return x
        


class ConvNeXt(nn.Module):
    def __init__(self,
                 block:Bottleneck,
                 depths:list[int],
                 widths:list[int],
                 inchannel:int=3,
                 outsize:tuple[int]=(30, 14, 14),
                 droppath:float=0.):
        super(ConvNeXt, self).__init__()
        self.norm_layer = partial(nn.InstanceNorm2d, eps=1e-5)

        outchannel, outheight, outwidth = outsize

        self.depths = depths
        self.widths = widths

        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(nn.Conv2d(inchannel, widths[0], kernel_size=4, stride=4),
                             self.norm_layer(widths[0]))
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(self.norm_layer(widths[i]),
                                             nn.Conv2d(widths[i], widths[i+1], kernel_size=2, stride=2))
            self.downsample_layers.append(downsample_layer)

        encoders = [self._make_layer(block, widths[i], depths[i], droppath=droppath) for i in range(len(depths))]
        self.encoders = nn.ModuleList(encoders)

        self.denet = self._make_detnet_layer(in_channels=widths[-1])
        self.pool = nn.AdaptiveAvgPool2d((outheight, outwidth))
        self.conv_end = nn.Conv2d(256, outchannel, 1)
        # self.norm_end = self.norm_layer(30)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            # elif isinstance(m, nn.InstanceNorm2d):
            #     m.weight.data.fill_(1)
            #     m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, droppath=0.):
        layers = [block(planes, planes, stride, groups=planes, droppath=droppath)]
        for _ in range(1, blocks):
            layers.append(block(planes, planes, groups=planes, droppath=droppath))
        return nn.Sequential(*layers)
    
    def _make_detnet_layer(self, in_channels):
        layers = [
            DetNet(in_planes=in_channels, planes=256, block_type='B'),
            DetNet(in_planes=256, planes=256, block_type='A'),
            DetNet(in_planes=256, planes=256, block_type='A')
        ]
        return nn.Sequential(*layers)

    def forward(self, x):
        for i in range(len(self.depths)):
            x = self.downsample_layers[i](x)
            x = self.encoders[i](x)
        x = self.denet(x)
        x = self.pool(x)
        x = self.conv_end(x)
        # x = self.norm_end(x)
        x = torch.sigmoid(x)
        x = x.permute(0, 2, 3, 1)  # (-1,14,14,30)

        return x



def convNext(depths:list[int], widths:list[int], **kwargs):
    """
    kwargs:
        inchannel:int = 3
        outsize:tuple[int] = (30, 14, 14)
        droppath:float = 0.
    Returns:
        ConvNeXt: ConvNeXt(depths, widths)
    """
    return ConvNeXt(Bottleneck, depths, widths, **kwargs)
     
def convNext_T(**kwargs):
    """
    kwargs:
        inchannel:int = 3
        outsize:tuple[int] = (30, 14, 14)
        droppath:float = 0.
    Returns:
        ConvNeXt: ConvNeXt-T(depths = [3, 3, 9, 3], widths = [96, 192, 384, 768])
    """
    return ConvNeXt(Bottleneck, [3, 3, 9, 3], [96, 192, 384, 768], **kwargs)

def convNext_S(**kwargs):
    """
    kwargs:
        inchannel:int = 3
        outsize:tuple[int] = (30, 14, 14)
        droppath:float = 0.
    Returns:
        ConvNeXt: ConvNeXt-S(depths = [3, 3, 27, 3], widths = [96, 192, 384, 768])
    """
    return ConvNeXt(Bottleneck, [3, 3, 27, 3], [96, 192, 384, 768], **kwargs)
    
def convNext_B(**kwargs):
    """
    kwargs:
        inchannel:int = 3
        outsize:tuple[int] = (30, 14, 14)
        droppath:float = 0.
    Returns:
        ConvNeXt: ConvNeXt-B(depths = [3, 3, 27, 3], widths = [128, 256, 512, 1024])
    """
    return ConvNeXt(Bottleneck, [3, 3, 27, 3], [128, 256, 512, 1024], **kwargs)
    
def convNext_L(**kwargs):
    """
    kwargs:
        inchannel:int = 3
        outsize:tuple[int] = (30, 14, 14)
        droppath:float = 0.
    Returns:
        ConvNeXt: ConvNeXt-L(depths = [3, 3, 27, 3], widths = [192, 384, 768, 1536])
    """
    return ConvNeXt(Bottleneck, [3, 3, 27, 3], [192, 384, 768, 1536], **kwargs)

def convNext_XL(**kwargs):
    """
    kwargs:
        inchannel:int = 3
        outsize:tuple[int] = (30, 14, 14)
        droppath:float = 0.
    Returns:
        ConvNeXt: ConvNeXt-XL(depths = [3, 3, 27, 3], widths = [256, 512, 1024, 2048])
    """
    return ConvNeXt(Bottleneck, [3, 3, 27, 3], [256, 512, 1024, 2048], **kwargs)