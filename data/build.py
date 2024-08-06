import numpy as np
import torch
from timm.data import Mixup

from .hsi_data import choose_sample
from .hsi_data import load_data, create_samples, pad_with_zeros, split_train_test_set, AllDataSet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
try:
    from torchvision.transforms import InterpolationMode
    def _pil_interp(method):
        if method == 'bicubic':
            return InterpolationMode.BICUBIC
        elif method == 'lanczos':
            return InterpolationMode.LANCZOS
        elif method == 'hamming':
            return InterpolationMode.HAMMING
        else:
            # default bilinear, do we want to allow nearest?
            return InterpolationMode.BILINEAR
    import timm.data.transforms as timm_transforms
    timm_transforms._pil_interp = _pil_interp
except:
    from timm.data.transforms import _pil_interp


# 构建数据集，swin_transformer和swin_mae微调都使用这个
def build_loader(config):
    ds = config.DATA.DATASET
    data, gt = load_data(ds)

    # data = log_normalize_image(data)  # 对数归一化
    # data = normalize_0_1(data)  # 0-1缩放
    # data = data[:,:,:150]   # TODO 3.13:IP取前150个光谱进行微调，再插值到200
    # data = interp_3d(data, 200)            # 插值
    # data = apply_guided_filter(data)
    # sample_weights = config.DATA.SAMPLE_WEIGHTS # 采样权重
    # pca = False                                 # pca变换
    fixed_sampling = True                       # 固定采样
    margin = config.DATA.MARGIN                 # 填充边距
    remove_zero = config.DATA.REMOVE_ZERO_LABELS# 移除未分类像素
    test_ratio = config.DATA.TEST_RATIO         # 测试比例
    # ***************************
    # L = len(config.MODEL.SWIN.DEPTHS)            # 网络层数
    # window_size = config.MODEL.SWIN.WINDOW_SIZE      # window_size
    # interp_dim = window_size*2**(L-1)           # 插值维度
    # # 预处理：线性插值
    # data = pre_process(data, interp_dim)
    # TODO 0-1缩放
    # data = normalize_0_1(data)              # 0-1归一化

    # data = interp_3d(data, 200)  # 插值



    # pca变换
    # if pca:
    #     data = apply_pca(data, 64)
    # 另一种数据集
    # if ds.endswith('_c'):
    #     gt[:, 0] = gt[:, 0] - 1
    #     dataset, labels = create_samples_corrected(pad_with_zeros(data, margin), gt, margin)
    #     labels = labels - 1
    #     labels = labels / 1.
    # else:
    dataset, labels = create_samples(pad_with_zeros(data, margin), gt, margin=margin, remove_zero_labels= not fixed_sampling)
    # del data
    # 固定采样
    if fixed_sampling:  # 如果加入未分类像素，train由选定样本加上0元素，nonzero变成所有元素
        train_poi, nonzero_poi, test_poi, weights = choose_sample(ds,50,15, remove_zero = remove_zero)
        train_data = dataset[train_poi,:,:,:]
        train_label = labels[train_poi,] - 1
        test_data = dataset[test_poi, :, :, :]
        test_label = labels[test_poi,] - 1
        dataset = dataset[nonzero_poi,:,:,:]
        labels = labels[nonzero_poi,] - 1


        # train_poi, val_poi, test_poi, weights = choose_sample4finetune(gt, 50, 15)
        # nonzero_poi = np.concatenate((train_poi, val_poi, test_poi))
        # train_data = dataset[train_poi, :, :, :]
        # train_label = labels[train_poi,] - 1
        #
        # val_data = dataset[val_poi, :, :, :]
        # val_label = labels[val_poi,] - 1
        #
        # test_data = dataset[test_poi, :, :, :]
        # test_label = labels[test_poi,] - 1
        #
        # dataset = dataset[nonzero_poi, :, :, :]
        # labels = labels[nonzero_poi,]

    else:
        train_data, test_data, train_label, test_label = split_train_test_set(dataset, labels, test_ratio, )
        # 损失函数的分类权重
        weights = np.bincount(np.int64(train_label))
        weights = max(weights) / weights

    # sampler_train = torch.utils.data.RandomSampler(train_data)
    # sampler_val = torch.utils.data.RandomSampler(test_data)
    # sampler_all = torch.utils.data.RandomSampler(dataset)

    # 样本权重
    # if sample_weights:
    #     w = torch.zeros(len(train_label))
    #     counter = Counter(train_label)
    #     for (i, x) in enumerate(train_label):
    #         w[i] = 1 / counter[x]
    #     sampler_train = torch.utils.data.WeightedRandomSampler(w, len(train_label))

    # sampler_train = SubsetRandomSampler(dataset_train)
    # sampler_val = SubsetRandomSampler(dataset_val)

    train_data = AllDataSet(train_data, train_label)
    # val_data = AllDataSet(val_data, val_label)
    val_data = AllDataSet(test_data, test_label)
    test_data = AllDataSet(test_data, test_label)
    dataset = AllDataSet(dataset, labels)

    dataloader_finetune = torch.utils.data.DataLoader(
        train_data,
        # sampler=sampler_train,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=True,
        shuffle = True
    )

    dataloader_val = torch.utils.data.DataLoader(
        val_data,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=True,
        shuffle=True
    )

    dataloader_test = torch.utils.data.DataLoader(
        test_data,
        # sampler=sampler_val,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False
    )

    dataloader_all = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False
    )

    # setup mixup / cutmix
    mixup_fn = None
    mixup_active = config.AUG.MIXUP > 0 or config.AUG.CUTMIX > 0. or config.AUG.CUTMIX_MINMAX is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=config.AUG.MIXUP, cutmix_alpha=config.AUG.CUTMIX, cutmix_minmax=config.AUG.CUTMIX_MINMAX,
            prob=config.AUG.MIXUP_PROB, switch_prob=config.AUG.MIXUP_SWITCH_PROB, mode=config.AUG.MIXUP_MODE,
            label_smoothing=config.MODEL.LABEL_SMOOTHING, num_classes=config.MODEL.NUM_CLASSES)

    return dataloader_finetune, dataloader_val, dataloader_test, mixup_fn, weights, dataloader_all, gt


# 构建Swin_MAE 预训练数据集 全部用于预训练
def build_loader_swin_mae(config):
    ds = config.DATA.DATASET
    data, gt = load_data(ds)

    # ***************************
    # L = len(config.MODEL.SWIN_MAE.DEPTHS)                 # 网络层数
    # window_size = config.MODEL.SWIN_MAE.WINDOW_SIZE       # window_size
    # interp_dim = window_size*2**(L-1)                     # 插值维度
    # data = pre_process(data, interp_dim=interp_dim)
    # TODO 0-1缩放
    # data = normalize_0_1(data)                            # 0-1缩放
    # data = data[:,:,:150]                                 # TODO 3.13:IP取前150个光谱进行预训练，再插值到200
    # data = interp_3d(data, 200)                           # 插值
    # data = log_normalize_image(data)                      # 对数归一化
    # remove_zero = config.DATA.REMOVE_ZERO_LABELS          # 移除未分类像素

    margin = config.DATA.MARGIN
    dataset, labels = create_samples(pad_with_zeros(data, margin), gt, margin=margin, remove_zero_labels=True)
    # dataset, labels = create_samples(pad_with_zeros(data, margin), gt, margin=margin, remove_zero_labels=False)
    # ----------------------------------------------------------------
    # train_poi, nonzero_poi, test_poi = choose_sample_for_pretrain(gt, 2000, 1000, remove_zero=True)
    # train_data = dataset[train_poi, :, :, :]    # train_poi:(18000,)
    # train_label = labels[train_poi,] - 1
    # dataset = train_data
    # labels = train_label
    # ----------------------------------------------------------------

    dataset = AllDataSet(dataset, labels)

    sampler = torch.utils.data.RandomSampler(dataset)

    # sampler_train = SubsetRandomSampler(dataset_train)
    # sampler_val = SubsetRandomSampler(dataset_val)

    data_loader = torch.utils.data.DataLoader(
        dataset.x_data,
        sampler=sampler,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=True,
    )
    return data_loader, gt