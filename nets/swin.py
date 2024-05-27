from torch import nn
from torchvision.models import SwinTransformer

from functools import partial

from config.swin_config import SwinTransformerVersion

class SwinTransformerHead(nn.Module):
    def __init__(self, inchannel: int, outchannel: int, outheight: int, outwidth: int):
        super(SwinTransformerHead, self).__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-5)
        self.conv1 = nn.Conv2d(inchannel, outchannel, 1)
        self.norm1 = norm_layer((outchannel, outheight, outwidth))

        # self.conv1 = nn.Conv2d(inchannel, 512, 3, 1, 2)
        # self.conv2 = nn.Conv2d(512, 256, 3, 1, 2)
        # self.conv3 = nn.Conv2d(256, 128, 3, 1, 2)
        # self.conv4 = nn.Conv2d(128, outchannel, 1)

        # self.norm1 = norm_layer((512, outheight, outwidth))
        # self.norm2 = norm_layer((256, outheight, outwidth))
        # self.norm3 = norm_layer((128, outheight, outwidth))
        # self.norm4 = norm_layer((outchannel, outheight, outwidth))
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation(x)

        return x



class SwinTransformerBasedYOLOv1(nn.Module):
    def __init__(self, cfg:dict, version:SwinTransformerVersion=None):
        super(SwinTransformerBasedYOLOv1, self).__init__()
        if version:
            backbone_cfg = cfg["BACKBONE"]["SWIN"][version]
            self.backbone = backbone_cfg["MODEL"](backbone_cfg["WEIGHTS"])
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
        x = x.permute(0, 2, 3, 1)
        return x



# SwinTransformer
def swintransformer(cfg, version=None):
    model = SwinTransformerBasedYOLOv1(cfg, version)
    return model