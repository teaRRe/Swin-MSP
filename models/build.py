from .SwinMSP import build_swin_msp
from .SFBT import SwinTransformer
import torch.nn as nn


def build_model(config, is_pretrain=False, **kwargs):
    model_type = config.MODEL.TYPE
    layernorm = nn.LayerNorm

    kwargs.setdefault('bn', False)
    kwargs.setdefault('cpe', False)
    kwargs.setdefault('num_patches', 200)   # 设置默认通道数,主文件传进来的会覆盖掉这个参数
    kwargs.setdefault('batch_size', config.DATA.BATCH_SIZE)
    kwargs.setdefault('residual', False)
    kwargs.setdefault('norm_targets', config.DATA.NORM_TARGETS)
    kwargs.setdefault('act', config.DATA.ACT)

    # 构建预训练的SwinMAE
    if is_pretrain:
        model = build_swin_msp(config, **kwargs)
        return model

    # 默认是SwinTransformer模型, 用于预训练的SwinMAE也是用SwinTransformer为骨架
    # 创建模型时需要根据通道数动态生成一些参数
    # 或者在patch_embed部分计算
    EMBED_DIM = config.MODEL.SWIN.EMBED_DIM
    if model_type == 'swin':
        model = SwinTransformer(patch_size=config.MODEL.SWIN.PATCH_SIZE,
                                in_chans=config.MODEL.SWIN.IN_CHANS,
                                num_classes=config.MODEL.NUM_CLASSES,
                                embed_dim=EMBED_DIM,
                                depths=config.MODEL.SWIN.DEPTHS,
                                num_heads=config.MODEL.SWIN.NUM_HEADS,
                                window_size=config.MODEL.SWIN.WINDOW_SIZE,
                                mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
                                qkv_bias=config.MODEL.SWIN.QKV_BIAS,
                                qk_scale=config.MODEL.SWIN.QK_SCALE,
                                drop_rate=config.MODEL.DROP_RATE,
                                drop_path_rate=config.MODEL.DROP_PATH_RATE,
                                ape=config.MODEL.SWIN.APE,
                                norm_layer=layernorm,
                                patch_norm=config.MODEL.SWIN.PATCH_NORM,
                                use_checkpoint=config.TRAIN.USE_CHECKPOINT,
                                fused_window_process=config.FUSED_WINDOW_PROCESS,
                                **kwargs)

    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model
