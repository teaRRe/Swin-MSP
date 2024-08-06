# --------------------------------------------------------
# UC3L (Uniaxial Continuous Cross Correlation Layer)
# --------------------------------------------------------
import math

from einops import rearrange
import torch.nn as nn
import torch
import torch.nn.functional as F


def norm_targets(targets, patch_size):
    assert patch_size % 2 == 1

    targets_ = targets
    targets_count = torch.ones_like(targets)

    targets_square = targets ** 2.
    # F.avg_pool2d 输入张量形状理应为(batch_size, channels, height, width)
    targets_mean = F.avg_pool2d(targets, kernel_size=patch_size, stride=1, padding=patch_size // 2,
                                count_include_pad=False)
    targets_square_mean = F.avg_pool2d(targets_square, kernel_size=patch_size, stride=1, padding=patch_size // 2,
                                       count_include_pad=False)
    targets_count = F.avg_pool2d(targets_count, kernel_size=patch_size, stride=1, padding=patch_size // 2,
                                 count_include_pad=True) * (patch_size ** 2)

    targets_var = (targets_square_mean - targets_mean ** 2.) * (targets_count / (targets_count - 1))
    targets_var = torch.clamp(targets_var, min=0.)

    targets_ = (targets_ - targets_mean) / (targets_var + 1.e-6) ** 0.5

    return targets_


def cpe(n, l):
    import numpy as np2
    pos = np2.zeros((n, l, l))
    xc = l // 2
    yc = l // 2
    for i in range(n):
        for h in range(l):
            for k in range(l):
                pos[i, h, k] = np2.max([np2.abs(h - xc), np2.abs(k - yc)]) + 1
    return pos


# 空间卷积与单层卷积
class PatchEmbedding(nn.Module):
    """
    空间卷积(kernel:(patch_size,patch_size) stride:(patch_size,patch_size))
    与单层卷积(kernel:(1, patch_size*5) stride:(1, patch_size*5))
    """

    def __init__(self, num_patches, in_channels, patch_size, embed_dim, norm_layer: nn.Module = None, **kwargs):
        super().__init__()
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        self.norm_targets = kwargs['norm_targets']
        act = kwargs['act']
        self.s_w = num_patches
        self.patches_resolution = (1, num_patches)
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.in_chans = in_channels
        self.act = nn.GELU() if act else nn.Identity()  # IP数据集
        # self.act = nn.Identity()  # 其他
        if num_patches % self.patch_size == 0:
            padding = (0, 0)
        else:
            padding = (0, (math.ceil(num_patches / self.patch_size) * self.patch_size - num_patches) * self.patch_size)
        self.proj = nn.Conv2d(1, out_channels=embed_dim, kernel_size=(patch_size, patch_size), stride=patch_size)
        self.proj_1 = nn.Conv2d(1, out_channels=embed_dim, kernel_size=(1, patch_size * patch_size),
                                stride=(1, patch_size * patch_size), padding=padding)
        self.proj_norm = nn.LayerNorm(num_patches)
        self.proj_1norm = nn.LayerNorm(int(num_patches / patch_size))

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size * self.patch_size)
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops

    def conv_spectral(self, x: torch.Tensor):
        x = x.unsqueeze(1)
        x = rearrange(x, "B C H W N -> B C H (W N)")  # B,1,p,p,n
        x = self.proj(x)
        # x = self.proj_norm(x)
        x = rearrange(x, "B L C N -> B C N L")  # -> 批量大小,通道,patch数量,映射维度
        return x

    def conv_spatial(self, x: torch.Tensor):  # B,5,5,200
        x_t = x.transpose(1, 2)

        x = x.unsqueeze(1)
        x = rearrange(x, "B C H W N -> B C H (W N)")  # B,1,5,1000
        x_spa1 = self.proj_1(x)  # 96,96,5,40
        # x_spa1 = self.proj_1norm(x_spa1)

        x_t = x_t.unsqueeze(1)
        x_t = rearrange(x_t, "B C H W N -> B C H (W N)")
        x_spa2 = self.proj_1(x_t)
        # x_spa2 = self.proj_1norm(x_spa2)

        x = x_spa1 + x_spa2

        x = rearrange(x, "B C X Y -> B C (X Y)")
        x = x.transpose(1, 2)
        x = x.unsqueeze(1)

        x = x[:, :, :self.s_w, :]

        return x

    def forward(self, x: torch.Tensor):
        if self.norm_targets:
            x = norm_targets(x, 3)
        x = self.conv_spectral(x) + self.conv_spatial(x)
        x = self.norm(x)
        x = self.act(x)  # 加入激活函数, 配置文件里的ACT,hou有用
        return x
