# --------------------------------------------------------
# 基础 Swin Transformer 模块 （已弃用）
# --------------------------------------------------------
import os
import time
import json
import random
import argparse
import datetime
import numpy as np
import platform

import torch
import torch.backends.cudnn as cudnn

from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy, AverageMeter, NativeScaler

from config import get_config
from models import build_model
from data import build_loader
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from logger import create_logger
from utils import load_checkpoint, load_pretrained, save_checkpoint, NativeScalerWithGradNormCount, auto_resume_helper, \
    reduce_tensor
from validate import hsi_validate
from torch.utils.tensorboard import SummaryWriter


def parse_option():
    parser = argparse.ArgumentParser('Swin Transformer training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, required=False, metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default='',
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--pretrained',
                        help='pretrained weight from checkpoint, could be imagenet22k pretrained weight')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--disable_amp', action='store_true', help='Disable pytorch amp')
    parser.add_argument('--amp-opt-level', type=str, choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used (deprecated!)')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')

    # distributed training
    parser.add_argument("--local_rank", type=int, default=0, help='local rank for DistributedDataParallel')

    # for acceleration
    parser.add_argument('--fused_window_process', action='store_true', default=True,
                        help='Fused window shift & window partition, similar for reversed part.')
    parser.add_argument('--fused_layernorm', action='store_true', help='Use fused layernorm.')
    # overwrite optimizer in config (*.yaml) if specified, e.g., fused_adam/fused_lamb
    parser.add_argument('--optim', type=str,
                        help='overwrite optimizer if provided, can be adamw/sgd/fused_adam/fused_lamb.')

    args, unparsed = parser.parse_known_args()
    config = get_config(args)

    return args, config


def main(config):
    train_data, test_data, dataloader_train, dataloader_test, mixup_fn, weights, dataloader_all, gt = build_loader(config)

    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    # 创建模型 将通道数传进模型
    num_patches = train_data.x_data.shape[3]
    model = build_model(config, num_patches=num_patches)
    logger.info(str(model))
    # assert False, '停止运行'
    # 总参数数量
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    if hasattr(model, 'flops'):
        flops = model.flops()
        logger.info(f"number of GFLOPs: {flops / 1e9}")

    model.cuda()
    model_without_ddp = model

    optimizer = build_optimizer(config, model)
    loss_scaler = NativeScalerWithGradNormCount()

    # 分类文件和日志的时间戳标志
    plat = platform.system().lower()
    if plat == 'windows':
        file_time = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())
    else:
        file_time = time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime())

    # 学习率自适应调整策略
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        lr_scheduler = build_scheduler(config, optimizer, len(dataloader_train) // config.TRAIN.ACCUMULATION_STEPS)
    else:
        lr_scheduler = build_scheduler(config, optimizer, len(dataloader_train))
    # 损失函数
    if config.AUG.MIXUP > 0.:
        criterion = SoftTargetCrossEntropy()
    elif config.MODEL.LABEL_SMOOTHING > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=config.MODEL.LABEL_SMOOTHING)
    else:
        criterion = torch.nn.CrossEntropyLoss()
    # 损失权重
    if config.MODEL.LOSS_WEIGHTS:
        weights = torch.HalfTensor(weights)
        criterion = torch.nn.CrossEntropyLoss(weight=weights.cuda())

    max_accuracy = 0.0
    # 自动恢复训练
    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT)
        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}")
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')
    cls_resport = ''
    # 恢复: 如果没指定 resume 的 pth 路径，则从保存的pth里拿最新那个
    if config.MODEL.RESUME:
        # 恢复时直接加载参数往下走
        max_accuracy = load_checkpoint(config, model_without_ddp, optimizer, lr_scheduler, loss_scaler, logger)
        # acc1, acc5, loss = validate(config, data_loader_val, model)
        # cls_resport = hsi_validate(logger, model, data_loader_val, file_time, dataloader_all, labels,
        #                            train_type='swin_transformer')
        # logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
        if config.EVAL_MODE:
            return
    if config.MODEL.PRETRAINED and (not config.MODEL.RESUME):
        load_pretrained(config, model_without_ddp, logger)
        acc1, acc5, loss = validate(config, dataloader_test, model)
        hsi_validate(config, logger, model, dataloader_test, file_time, dataloader_all, gt,
                     train_type='swin_transformer')
        logger.info(f"Accuracy of the network on the {len(test_data)} test images: {acc1:.1f}%")
    # 吞吐率
    if config.THROUGHPUT_MODE:
        throughput(dataloader_test, model, logger)
        return

    logger.info("Start training")
    start_time = time.time()

    # 日志文件位置，以时间戳命名文件夹
    logpath = r'./log/swin_transformer/'
    if not os.path.exists(logpath):
        os.makedirs(logpath)
    # --------------------------------------------------------------
    # 恢复训练：
    # 1. 如果在配置文件中指定auto_resume为true，则拿取保存的最新的pth文件进行训练
    # 2. 也可以在运行配置中指定resume路径，就是pth文件的路径接着训练
    # --------------------------------------------------------------
    if not config.TRAIN.AUTO_RESUME:
        # os.makedirs(logpath + file_time)
        writer = SummaryWriter(logpath + file_time + config.TAG if config.TAG else '')
    else:  # 恢复训练结果的日志单独命名
        # os.makedirs(logpath + file_time + "_RESUME")
        writer = SummaryWriter(logpath + file_time + "_RESUME" + config.TAG if config.TAG else '')

    # 将模型和参数写入每次训练日志
    writer.add_text("config", "<pre>" + config.dump() + "</pre>")
    writer.add_text("model", "<pre>" + str(model) + "</pre>")

    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        # data_loader_train.sampler.set_epoch(epoch)

        # 训练一个epoch
        loss = train_one_epoch(config, model, criterion, dataloader_train,
                               optimizer, epoch, mixup_fn, lr_scheduler, loss_scaler, writer)

        # loss可视化
        writer.add_scalar('swin_transformer_loss', loss, epoch)

        # 保存模型，SAVE_FREQ 的倍数和最后一次训练
        if epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1):
            save_checkpoint(config, epoch, model_without_ddp, max_accuracy, optimizer, lr_scheduler, loss_scaler,
                            logger, loss, file_time)
        acc1, _, loss = validate(config, dataloader_test, model)
        # acc1可视化
        writer.add_scalar('swin_transformer_acc', acc1, epoch)

    # Swin_Transformer分类验证
    # acc1, acc5, loss = validate(config, data_loader_val, model)
    # logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
    # max_accuracy = max(max_accuracy, acc1)
    # logger.info(f'Max accuracy: {max_accuracy:.2f}%')
    logger.info(r'训练结束，开始测试')

    # HSI验证方法验证
    cls_resport, pre_data, gt_data = hsi_validate(config, logger, model, dataloader_test, file_time, dataloader_all,
                                                  gt,
                                                  train_type='swin_transformer')
    # 分类结果写入日志
    writer.add_text("cls_result", "<pre>" + cls_resport + "</pre>")
    writer.add_image('gt', gt_data)
    writer.add_image('prediction', pre_data)

    writer.close()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))


def train_one_epoch(config, model, criterion, data_loader, optimizer, epoch, mixup_fn, lr_scheduler, loss_scaler,
                    writer):
    model.train()
    optimizer.zero_grad()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()
    scaler_meter = AverageMeter()

    start = time.time()
    end = time.time()

    for i, (samples, targets) in enumerate(data_loader):
        samples = samples.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        # mixup 数据增强
        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        # 混合精度计算
        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            outputs = model(samples)
        loss = criterion(outputs, targets)
        loss = loss / config.TRAIN.ACCUMULATION_STEPS

        # logger.info(f'loss value = {loss}, is nan? {math.isnan(loss)}')

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(loss, optimizer, clip_grad=config.TRAIN.CLIP_GRAD,
                                parameters=model.parameters(), create_graph=is_second_order,
                                update_grad=(i + 1) % config.TRAIN.ACCUMULATION_STEPS == 0)
        if (i + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
            optimizer.zero_grad()
            lr_scheduler.step_update((epoch * num_steps + i) // config.TRAIN.ACCUMULATION_STEPS)
        loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        loss_meter.update(loss.item(), targets.size(0))
        if grad_norm is not None:  # loss_scaler return None if not update
            norm_meter.update(grad_norm)
        scaler_meter.update(loss_scale_value)
        batch_time.update(time.time() - end)
        end = time.time()

        # if i % config.PRINT_FREQ == 0:
        #     lr = optimizer.param_groups[0]['lr']
        #     wd = optimizer.param_groups[0]['weight_decay']
        #     memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
        #     etas = batch_time.avg * (num_steps - i)
        #     logger.info(
        #         f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{i}/{num_steps}]\t'
        #         f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t wd {wd:.4f}\t'
        #         f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
        #         f'loss {loss_meter.val:.6f} ({loss_meter.avg:.4f})\t'
        #         f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
        #         f'loss_scale {scaler_meter.val:.4f} ({scaler_meter.avg:.4f})\t'
        #         f'mem {memory_used:.0f}MB')
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}  loss:{loss}")
    return loss


@torch.no_grad()
def validate(config, data_loader, model):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    end = time.time()
    for idx, (images, target) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            output = model(images)

        # measure accuracy and record loss
        loss = criterion(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        acc1 = reduce_tensor(acc1)
        acc5 = reduce_tensor(acc5)
        loss = reduce_tensor(loss)

        loss_meter.update(loss.item(), target.size(0))
        acc1_meter.update(acc1.item(), target.size(0))
        acc5_meter.update(acc5.item(), target.size(0))

        # 测量运行时间
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(
                # f'Test: [{idx}/{len(data_loader)}]\t'
                # f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Loss {loss_meter.val:.6f} ({loss_meter.avg:.6f})\t'
                f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
                # f'Acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f})\t'
                # f'Mem {memory_used:.0f}MB'
            )
    logger.info(f' * Acc@1 {acc1_meter.avg:.3f}')
    return acc1_meter.avg, acc5_meter.avg, loss_meter.avg


@torch.no_grad()
def throughput(data_loader, model, logger):
    model.eval()

    for idx, (images, _) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        batch_size = images.shape[0]
        for i in range(50):
            model(images)
        torch.cuda.synchronize()
        logger.info(f"throughput averaged with 30 times")
        tic1 = time.time()
        for i in range(30):
            model(images)
        torch.cuda.synchronize()
        tic2 = time.time()
        logger.info(f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}")
        return


if __name__ == '__main__':
    args, config = parse_option()

    # if config.AMP_OPT_LEVEL:
    #     print("[warning] Apex amp has been deprecated, please use pytorch amp instead!")

    seed = config.SEED
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    # 根据总批量大小线性缩放学习率，可能不是最佳的
    s = 128.
    linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE / s
    linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE / s
    linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE / s
    # 梯度积累也需要缩放学习率
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_warmup_lr = linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS
    config.defrost()
    # 覆盖配置文件中的学习率参数
    config.TRAIN.BASE_LR = linear_scaled_lr
    config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    config.TRAIN.MIN_LR = linear_scaled_min_lr
    config.freeze()

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=0, name=f"{config.MODEL.NAME}")

    logger.info(config.dump())
    logger.info(json.dumps(vars(args)))

    main(config)
