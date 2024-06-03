import torch
import math

from torch import nn

from nets.nn import Bottleneck, DetNet
        
class NextBottleneck(Bottleneck):
    expansion = 2
    def __init__(self, in_planes, planes, stride=1, downsample=None, groups=32):
        super(NextBottleneck, self).__init__(in_planes, planes, stride, downsample, groups)
    
    def forward(self, x):
        return super(NextBottleneck, self).forward(x)



class ResNeXt(nn.Module):
    def __init__(self, block, layers, cardinality=32, num_classes=1000):
        self.in_planes = 64
        super(ResNeXt, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 128, layers[0], cardinality)
        self.layer2 = self._make_layer(block, 256, layers[1], cardinality, stride=2)
        self.layer3 = self._make_layer(block, 512, layers[2], cardinality, stride=2)
        self.layer4 = self._make_layer(block, 1024, layers[3], cardinality, stride=2)
        self.layer5 = self._make_detnet_layer(in_channels=2048)

        self.conv_end = nn.Conv2d(256, 30, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_end = nn.BatchNorm2d(30)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, cardinality, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        #[3, 4, 6, 3]
        layers = [block(self.in_planes, planes, stride, downsample, groups=cardinality)]
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes, groups=cardinality))

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
    


# resnext50
def resnext50(pretrained=False, **kwargs):
    model = ResNeXt(NextBottleneck, [3, 4, 6, 3], **kwargs)
    return model



# resnext50
def resnext152(pretrained=False, **kwargs):
    model = ResNeXt(NextBottleneck, [3, 8, 36, 3], **kwargs)
    return model