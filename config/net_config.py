from .swin_config import SWIN_CONFIG
from .vit_config import VIT_CONFIG

NET_CONFIG = {
    "BACKBONE": {
        "SWIN": SWIN_CONFIG,
        "VIT": VIT_CONFIG,
    },
    "INCHANNEL": 3,
    "OUTCHANNEL": 30,
    "OUTHEIGHT": 14,
    "OUTWIDTH": 14,
    "TRAIN_WEIGHTS": {
        "SWIN": ["head"]
    }
}


