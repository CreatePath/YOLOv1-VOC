import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

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
        self.activation = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.activation(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.activation(out)

        return out



class DivConv(nn.Module):
    '''
    yolo는 5:5:20의 비율로 bbox, bbox, classification prediction을 수행
    해당 비율을 고려하여 설계함.
    '''
    expansion = 2
    def __init__(self, in_planes:int, planes:int, ratio:tuple[int]) -> None:
        super(DivConv, self).__init__()

        sum_ratio = sum(ratio)
        self.inplanes_list = []
        self.convlist = nn.ModuleList()
        for r in ratio:
            inplanes_r = in_planes * r // sum_ratio
            planes_r = planes * r // sum_ratio
            self.inplanes_list.append(inplanes_r)
            self.convlist.append(nn.Sequential(
                nn.Conv2d(inplanes_r, planes_r*self.expansion, kernel_size=3, padding=1),
                nn.BatchNorm2d(planes_r * self.expansion),
                nn.ReLU(),
                nn.Conv2d(planes_r * self.expansion, planes_r, kernel_size=3, padding=1),
                nn.BatchNorm2d(planes_r)
            ))

        self.conv3 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, groups=planes)
        self.bn3 = nn.BatchNorm2d(planes)

        self.downsample = None
        if in_planes != planes:
            self.downsample = nn.Conv2d(in_planes, planes, 3, 1, 1)
        # self.activation = nn.GELU()
        
    def forward(self, x):
        residual = x
        start = 0
        splited_x = []
        for inplanes in self.inplanes_list:
            splited_x.append(x[:, start:start+inplanes])
            start += inplanes
        
        for i in range(len(splited_x)):
            splited_x[i] = self.convlist[i](splited_x[i])

        out = torch.concat(splited_x, dim=1)
        out = self.conv3(out)

        if self.downsample:
            residual = self.downsample(x)

        out = out + residual
        out = self.bn3(out)
        # out = self.activation(out)

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
        self.activation = nn.ReLU()

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.activation(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.downsample(x)
        out = self.activation(out)
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
        self.layer5 = self._make_detnet_layer(in_channels=2048)
        # self.conv_end = nn.Conv2d(256, 30, 3, 1, 1, bias=False)
        # self.bn_end = nn.BatchNorm2d(30)
        # self.conv_end = nn.Sequential(nn.Conv2d(256, 30, 3, 1, 1, bias=False),
        #                               *[DivConv
        #                             (30, 30) for _ in range(3)])
        self.dropout = nn.Dropout2d(0.1)

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
        layers = [block(self.in_planes, planes, stride, downsample)]
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def _make_detnet_layer(self, in_channels):
        layers = [
            DetNet(in_planes=in_channels, planes=384, block_type='B'),
            DetNet(in_planes=384, planes=384, block_type='A'),
            DetNet(in_planes=384, planes=384, block_type='A'),
            DivConv(384, 192, (1, 1, 4)),
            DivConv(192, 96, (1, 1, 4)),
            DivConv(96, 30, (1, 1, 4))
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
        
        # x = self.conv_end(x)
        # x = self.bn_end(x)
        x = self.dropout(x)
        x = torch.sigmoid(x)
        x = x.permute(0, 2, 3, 1)

        return x



# resnet50
def resnet50(pretrained=False, **kwargs):
    model_ = ResNet(Bottleneck, [3, 8, 16, 3], **kwargs)
    # if pretrained:
    #     model_.load_state_dict(model_zoo.load_url(resnet152_url))
    return model_

# resnet152
def resnet152(pretrained=False, **kwargs):
    model_ = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    # if pretrained:
    #     model_.load_state_dict(model_zoo.load_url(resnet152_url))
    return model_


if __name__ == '__main__':
    a = torch.randn((8, 3, 448, 448))
    model = resnet152()
    print(model(a).shape)
