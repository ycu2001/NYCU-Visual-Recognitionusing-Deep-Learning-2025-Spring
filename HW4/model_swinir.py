# -----------------
# SwinIR Pytorch Implementation (lite version for CV HW4)
# -----------------
import torch
import torch.nn as nn
import torch.nn.functional as F

def to_2tuple(x):
    return (x, x)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, groups=hidden_features)
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class LayerNorm2d(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        x = (x - mu) / (var + self.eps).sqrt()
        return self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3) * x + self.bias.unsqueeze(0).unsqueeze(2).unsqueeze(3)

class SwinBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm1 = LayerNorm2d(dim)
        self.attn = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.norm2 = LayerNorm2d(dim)
        self.mlp = Mlp(dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class SwinIR(nn.Module):
    def __init__(self, upscale=1, in_chans=3, img_size=64, window_size=8,
                 img_range=1.0, depths=[6, 6, 6, 6], embed_dim=96,
                 num_heads=[6, 6, 6, 6], mlp_ratio=2., upsampler='', resi_connection='1conv'):
        super().__init__()
        self.conv_first = nn.Conv2d(in_chans, embed_dim, 3, 1, 1)
        self.blocks = nn.Sequential(*[SwinBlock(embed_dim) for _ in range(12)])  # simple 12 layer
        self.conv_last = nn.Conv2d(embed_dim, in_chans, 3, 1, 1)

    def forward(self, x):
        x = self.conv_first(x)
        x = self.blocks(x)
        x = self.conv_last(x)
        return x
