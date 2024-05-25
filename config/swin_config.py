from enum import Enum
from torchvision.models import Swin_T_Weights, Swin_B_Weights, Swin_S_Weights
from torchvision.models import swin_t, swin_b, swin_s

class SwinTransformerVersion(Enum):
    SWIN_B = "SWIN_B"
    SWIN_T = "SWIN_T"
    SWIN_S = "SWIN_S"

SWIN_CONFIG = {
    SwinTransformerVersion.SWIN_B: {
        "MODEL": swin_b(Swin_B_Weights),
        "EMBED_DIM": 128,
    },
    SwinTransformerVersion.SWIN_S: {
        "MODEL": swin_s(Swin_S_Weights),
        "EMBED_DIM": 96,
    },
    SwinTransformerVersion.SWIN_T: {
        "MODEL": swin_t(Swin_T_Weights),
        "EMBED_DIM": 96,
    },
    "CUSTOM": {
        "PATCH_SIZE": [4, 4],
        "EMBED_DIM": 128,
        "DEPTHS": [2, 2, 18, 2],
        "NUM_HEADS": [4, 8, 16, 32],
        "WINDOW_SIZE": [7, 7],
        "STOCHASTIC_DEPTH_PROB": 0.5,
        "WIEHGTS": None,
        "PROGRESS": None
    }
}