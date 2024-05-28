import torch
from torch import nn

from functools import partial, reduce
from einops import rearrange

class PatchEmbedding(nn.Module):
    def __init__(self, patch_size:tuple[int], patch_dim:int, dim:int) -> None:
        super(PatchEmbedding, self).__init__()
        self.patch_height, self.patch_width = patch_size

        norm_layer = partial(nn.LayerNorm, eps=-1e-5)
        self.norm1 = norm_layer(patch_dim)
        self.mlp = nn.Linear(patch_dim, dim)
        self.norm2 = norm_layer(dim)

    def forward(self, x):
        x = rearrange(x,
                      'b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                      p1=self.patch_height,
                      p2=self.patch_width)
        x = self.norm1(x)
        x = self.mlp(x)
        x = self.norm2(x)
        return x

 

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim:int, num_heads:int=8, dropout:float=0.0) -> None:
        super(MultiHeadSelfAttention, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** (-0.5)
        
        self.norm = nn.LayerNorm(dim, eps=1e-5)

        self.query = nn.Linear(dim, dim, bias=False)
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)

        self.softmax = nn.Softmax(dim=-1)
        self.out = nn.Linear(dim, dim)

        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        B, N, C = x.size()
        x = self.norm(x)

        q, k, v = self.query(x), self.key(x), self.value(x)

        q = q.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        q *= self.scale
        qk = q @ k.transpose(-1, -2)
        attn = self.dropout(self.softmax(qk))
        out = attn @ v

        out = out.transpose(1, 2).contiguous().view(B, N, C)
        out = self.dropout(self.out(out))

        return out



class FeedForward(nn.Module):
    def __init__(self, dim:int, hidden_dim:int, dropout:float=0.0) -> None:
        super(FeedForward, self).__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim, eps=1e-5)
        self.mlp1 = nn.Linear(dim, hidden_dim)
        self.mlp2 = nn.Linear(hidden_dim, dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.norm(x)
        x = self.mlp1(x)
        x = self.activation(x)
        x = self.mlp2(x)
        x = self.dropout(x)
        return x



class TransformerEncoderBlock(nn.Module):
    def __init__(self, dim:int, num_heads:int=8, mlp_hidden_dim:int=3072, dropout:float=0.0) -> None:
        super(TransformerEncoderBlock, self).__init__()
        self.attention = MultiHeadSelfAttention(dim, num_heads, dropout)
        self.ffn = FeedForward(dim, mlp_hidden_dim, dropout)
    
    def forward(self, x):
        x = self.attention(x) + x
        x = self.ffn(x) + x
        return x



class VisionTransformer(nn.Module):
    def __init__(self,
                 image_size: tuple[int],
                 patch_size:tuple[int],
                 num_layers:int,
                 num_heads:int,
                 dim:int,
                 mlp_hidden_dim:int,
                 out_size:tuple[int],
                 dropout:float=0.0) -> None:
        super(VisionTransformer, self).__init__()
        channels, image_height, image_width = image_size
        patch_height, patch_width = patch_size
        self.out_size = out_size

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        encoder_list = [TransformerEncoderBlock(dim,
                                                num_heads,
                                                mlp_hidden_dim,
                                                dropout) for _ in range(num_layers)]

        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, dim))
        self.patch_embedding = PatchEmbedding(patch_size, patch_dim, dim)
        self.encoder = nn.Sequential(*encoder_list)
        
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(dim, reduce(lambda x, y: x * y, out_size))
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.patch_embedding(x)
        x += self.pos_embed

        x = self.dropout(x)

        x = self.encoder(x)
        x = x.mean(dim=1)
        x = self.out(x)
        x = self.relu(x)
        x = x.view(-1, *self.out_size).permute(0, 2, 3, 1) # (-1, 14, 14, 30)
        return x



def visionTransformer(vit_config:dict):
    return VisionTransformer(vit_config["IMAGE_SIZE"],
                              vit_config["PATCH_SIZE"],
                              vit_config["NUM_LAYERS"],
                              vit_config["NUM_HEADS"],
                              vit_config["DIM"],
                              vit_config["MLP_HIDDEN_DIM"],
                              vit_config["OUT_SIZE"],
                              vit_config["DROPOUT"])