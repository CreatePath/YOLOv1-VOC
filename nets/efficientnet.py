import torch

from torch import nn
from torchvision.models import EfficientNet

class EfficientNetBasedYOLOv1(nn.Module):
    def __init__(self, in_channels, version=None):
        super(EfficientNetBasedYOLOv1, self).__init__()
        self.network = EfficientNet.from_pretrained(version, in_channels=in_channels)
        self.output_layer = nn.Linear(1000, 26)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.network(x)
        x = self.relu(x)
        return x