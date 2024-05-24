from torchvision.models import Swin_T_Weights, Swin_B_Weights, Swin_S_Weights
from torchvision.models import swin_t, swin_b, swin_s

SWIN_CONFIG = {
    "SWIN_T": swin_t(Swin_T_Weights),
    "SWIN_B": swin_b(Swin_B_Weights),
    "SWIN_S": swin_s(Swin_S_Weights),
}