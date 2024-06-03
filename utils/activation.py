import torch

def stable_sigmoid(x):
    pos_mask = (x >= 0)
    neg_mask = (x < 0)
    
    z = torch.zeros_like(x)
    z[pos_mask] = torch.exp(-x[pos_mask])
    z[neg_mask] = torch.exp(x[neg_mask])
    
    top = torch.ones_like(x)
    top[neg_mask] = z[neg_mask]
    
    return top / (1 + z)