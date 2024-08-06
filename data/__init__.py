from .build import build_loader as _build_loader
from .build import build_loader_swin_mae
# from .data_simmim_ft import build_loader_finetune


def build_loader(config, swin_mae=False, is_pretrain=False):
    if not swin_mae:    # 默认构建用于训练的数据集
        return _build_loader(config)
    if is_pretrain:     # 构建用于预训练的数据集
        return build_loader_swin_mae(config)
    else:
        return _build_loader(config)
