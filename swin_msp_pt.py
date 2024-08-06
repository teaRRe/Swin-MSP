# --------------------------------------------------------
# SwinMSP 预训练模块
# --------------------------------------------------------
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import time
import argparse
import datetime
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.cuda.amp as amp
from timm.utils import AverageMeter
from tqdm import trange
from config import get_config
from models import build_model
from data import build_loader
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from logger import create_logger
from utils_simmim import load_checkpoint, save_checkpoint, get_grad_norm, auto_resume_helper
from torch.utils.tensorboard import SummaryWriter




def parse_option():
    parser = argparse.ArgumentParser('SimMIM pre-training script', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch_size', type=int, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--enable-amp', action='store_true')
    parser.add_argument('--disable-amp', action='store_false', dest='enable_amp')
    parser.set_defaults(enable_amp=True)
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')

    # distributed training
    parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')
    args = parser.parse_args()
    config = get_config(args)
    return args, config


def main(config):
    data_loader_train, gt = build_loader(config, swin_mae=True, is_pretrain=True)

    length = len(data_loader_train)
    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")

    num_patches = data_loader_train.dataset.shape[3]
    shape = data_loader_train.dataset.shape  # [N,p,p,C]

    model = build_model(config, is_pretrain=True, num_patches=num_patches, shape=shape)
    model.cuda()
    logger.info(str(model))

    optimizer = build_optimizer(config, model, simmim=True, is_pretrain=True)
    model_without_ddp = model

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    if hasattr(model_without_ddp, 'flops'):
        flops = model_without_ddp.flops()
        logger.info(f"number of GFLOPs: {flops / 1e9}")

    lr_scheduler = build_scheduler(config, optimizer, length)
    scaler = amp.GradScaler()

    # 分类文件和日志的时间戳标志
    file_time = time.strftime('%Y-%m-%d_%H_%M_%S', time.localtime())

    # 自动恢复训练
    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT, logger)
        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}")
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')

    # 手动恢复训练
    if config.MODEL.RESUME:
        load_checkpoint(config, model_without_ddp, optimizer, lr_scheduler, scaler, logger)

    logger.info("Start training")
    start_time = time.time()

    logpath = r'./log/swin_msp_pt/'
    if not os.path.exists(logpath):
        os.makedirs(logpath)
    writer = SummaryWriter(logpath + file_time + "_PRETRAIN_" + config.TAG if config.TAG else '')

    for epoch in trange(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS, leave=False, postfix=config.TAG):

        train_one_epoch(config, model, data_loader_train, optimizer, epoch, lr_scheduler, scaler, writer)
        if epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1):
            save_checkpoint(config, epoch, model_without_ddp, 0., optimizer, lr_scheduler, scaler, logger)

    # 将模型和参数写入每次训练日志
    writer.add_text("config", "<pre>" + config.dump() + "</pre>")
    writer.add_text("model", "<pre>" + str(model) + "</pre>")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))


def train_one_epoch(config, model, data_loader, optimizer, epoch, lr_scheduler, scaler, writer, **kwargs):
    model.train()
    optimizer.zero_grad()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()
    loss_scale_meter = AverageMeter()

    start = time.time()
    end = time.time()

    for idx, x in enumerate(data_loader):
        x = x.cuda(non_blocking=False)

        with amp.autocast(enabled=config.ENABLE_AMP):
            loss = model(x)

        if config.TRAIN.ACCUMULATION_STEPS > 1:
            loss = loss / config.TRAIN.ACCUMULATION_STEPS
            scaler.scale(loss).backward()
            if config.TRAIN.CLIP_GRAD:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
            else:
                grad_norm = get_grad_norm(model.parameters())
            if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
                scaler.step(optimizer)
                optimizer.zero_grad()
                scaler.update()
                lr_scheduler.step_update(epoch * num_steps + idx)
        else:
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            if config.TRAIN.CLIP_GRAD:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
            else:
                grad_norm = get_grad_norm(model.parameters())
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step_update(epoch * num_steps + idx)

        # t4 = time.time()
        # logger.info(f'ACCUMULATION_STEPS {t4-t3:.4f}s')

        torch.cuda.synchronize()

        loss_meter.update(loss.item(), x.size(0))
        norm_meter.update(grad_norm)
        loss_scale_meter.update(scaler.get_scale())
        batch_time.update(time.time() - end)
        end = time.time()

        writer.add_scalar('pretrain_loss', loss, epoch)
        lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('pretrain_lr', lr, epoch)

        if idx % config.PRINT_FREQ == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.8f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val:.8f} ({loss_meter.avg:.4f})\t'
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'loss_scale {loss_scale_meter.val:.2f} ({loss_scale_meter.avg:.2f})\t'
                f'mem {memory_used:.0f}MB')

    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")
    return loss


if __name__ == '__main__':
    _, config = parse_option()

    torch.cuda.set_device(config.LOCAL_RANK)
    seed = config.SEED
    torch.manual_seed(seed)
    np.random.seed(seed)
    # cudnn.benchmark = True
    cudnn.deterministic = True

    # 根据总批量大小线性缩放学习率，可能不是最佳的
    s = config.LR_SCALED
    linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE / s
    linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE / s
    linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE / s
    # 梯度积累也需要缩放学习率
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_warmup_lr = linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS
    config.defrost()
    config.TRAIN.BASE_LR = linear_scaled_lr
    config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    config.TRAIN.MIN_LR = linear_scaled_min_lr
    config.freeze()

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, name=f"{config.MODEL.NAME}")

    logger.info(config.dump())

    main(config)
