# --------------------------------------------------------
# SwinMSP (Shifted Windows Masked Spectral Pretraining Model)
# --------------------------------------------------------

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import trunc_normal_

from .SFBT import SwinTransformer
from .UC3L import norm_targets


def reverse(x: torch.Tensor, scale: int = 2) -> torch.Tensor:
    # x: 64,1,25,768
    B, H, W, C = x.shape
    x = x.view(B, H, -1, C // scale)
    return x


def window_masking4(mask_token, mask_ratio, x: torch.Tensor):
    """
    全局随机遮盖 mt:(1,1,96) x:(128,1,100,96) mr 0.75
    :param mask_token: 用来替换mask的可学习变量
    :param mask_ratio: mask比率
    :param x:
    :return:
    """
    x = rearrange(x, 'B H W C -> B (H W) C')
    B, L, D = x.shape  # 128,100,96

    # 生成随机噪声
    noise = np.random.rand(B, L)
    # 对噪声进行排序
    sparse_shuffle = np.argsort(noise, axis=1)
    # 获取前int(L * mask_ratio)个索引
    mask_index = sparse_shuffle[:, :int(L * mask_ratio)]
    keep_index = np.zeros((B, L - int(L * mask_ratio)), dtype=np.int64)

    for i in range(B):
        keep_index[i, :] = np.setdiff1d(sparse_shuffle[i], mask_index[i])

    x_masked = x.clone()
    x_masked[np.repeat(np.arange(B), mask_index.shape[1]), mask_index.flatten(), :] = mask_token
    mask = torch.ones_like(x_masked[:, :, 0])
    mask[np.repeat(np.arange(B), keep_index.shape[1]), keep_index.flatten()] = 0

    x_masked = rearrange(x_masked, 'B (H W) C -> B H W C', H=1)

    return x_masked, mask


class SFBTForSwinMSP(SwinTransformer):
    """
    以 SwinTransformer 为父类构建 SwinMSP 编码器，继承了SwinTransformer的所有字段。
    SwinTransformer 中定义的方法也会悉数继承，但是这里并不会去使用。
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # SwinMAE独有参数
        self.mask_token = nn.Parameter(torch.ones(1, 1, self.embed_dim))
        self.decoder_pred = nn.Linear(self.embed_dim, self.patch_size ** 2 * self.in_chans, bias=True)

        # 初始化权重
        trunc_normal_(self.mask_token, mean=0., std=.02)

        # 残差，默认开
        self.residual = True if not kwargs['residual'] else kwargs['residual']

    def forward(self, x):  # [B,H,W,C]
        x = self.patch_embed(x)  # patch_embed:[B,h,w,dim]
        x, mask = window_masking4(self.mask_token, self.mask_ratio, x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        if self.residual:
            outputs = []
            for i, layer in enumerate(self.layers):
                outputs.append(x)
                x = layer(x, outputs)
        else:
            for i, layer in enumerate(self.layers):
                x = layer(x)
        x = self.norm(x)
        return x, mask

    @torch.jit.ignore
    def no_weight_decay(self):
        return super().no_weight_decay() | {'mask_token'}


class SwinMSP(nn.Module):
    def __init__(self, config, encoder, encoder_stride, in_chans, patch_size):
        super().__init__()
        self.config = config
        self.encoder = encoder
        self.encoder.mask_ratio = config.MODEL.SWIN_MAE.MASK_RATIO
        self.encoder_stride = encoder_stride

        self.in_chans = in_chans
        self.patch_size = patch_size

        self.num_layers = self.encoder.num_layers
        self.embed = self.encoder.embed_dim
        self.num_patches = self.encoder.num_patches
        self.linear_rollback = nn.Linear(in_features=int(self.num_patches / 2 ** (self.num_layers - 1)),
                                         out_features=self.num_patches)  # 将merging到底层的窗口还原回原通道数量
        self.linear_decoder = nn.Linear(in_features=self.embed * 2 ** (self.num_layers - 1),
                                        out_features=self.patch_size ** 2)  # 将扩大到顶层的dim还原回原patch大小

    def forward_rec(self, x):
        z, mask = self.encoder(x)  # z:96,1,25,768 mask:96,200 x:96,5,5,200 z经过了norm

        z = z.squeeze(1).transpose(1, 2)
        z = self.linear_rollback(z)
        z = z.transpose(1, 2)
        z = self.linear_decoder(z)
        z = z.transpose(1, 2)

        x = norm_targets(x, 5)  # x:B,5,5,200
        x = rearrange(x, "B h w C -> B (h w) C")  # B 25 200
        loss_recon = F.mse_loss(x, z, reduction='none')
        loss_recon = loss_recon.mean(dim=1)
        loss = (loss_recon * mask).sum() / (mask.sum() + 1e-5) / self.in_chans

        return loss

    def forward(self, x):
        loss = self.forward_rec(x)
        return loss

    @torch.jit.ignore
    def no_weight_decay(self):
        if hasattr(self.encoder, 'no_weight_decay'):
            return {'encoder.' + i for i in self.encoder.no_weight_decay()}
        return {}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        if hasattr(self.encoder, 'no_weight_decay_keywords'):
            return {'encoder.' + i for i in self.encoder.no_weight_decay_keywords()}
        return {}


def build_swin_msp(config, **kwargs):
    model_type = config.MODEL.TYPE
    EMBED_DIM = config.MODEL.SWIN_MAE.EMBED_DIM
    if model_type == 'swin_mae':
        reconstruct_encoder = SFBTForSwinMSP(
            patch_size=config.MODEL.SWIN_MAE.PATCH_SIZE,
            in_chans=config.MODEL.SWIN_MAE.IN_CHANS,
            num_classes=config.MODEL.NUM_CLASSES,
            embed_dim=EMBED_DIM,
            depths=config.MODEL.SWIN_MAE.DEPTHS,
            num_heads=config.MODEL.SWIN_MAE.NUM_HEADS,
            window_size=config.MODEL.SWIN_MAE.WINDOW_SIZE,
            mlp_ratio=config.MODEL.SWIN_MAE.MLP_RATIO,
            qkv_bias=config.MODEL.SWIN_MAE.QKV_BIAS,
            qk_scale=config.MODEL.SWIN_MAE.QK_SCALE,
            drop_rate=config.MODEL.DROP_RATE,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            ape=config.MODEL.SWIN_MAE.APE,
            patch_norm=config.MODEL.SWIN_MAE.PATCH_NORM,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT,
            **kwargs)

        encoder_stride = 5
        in_chans = config.MODEL.SWIN_MAE.IN_CHANS
        patch_size = config.MODEL.SWIN_MAE.PATCH_SIZE
    else:
        raise NotImplementedError(f"Unknown pre-train model: {model_type}")

    model = SwinMSP(config=config, encoder=reconstruct_encoder, encoder_stride=encoder_stride, in_chans=in_chans,
                    patch_size=patch_size)

    return model
