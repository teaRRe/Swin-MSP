# --------------------------------------------------------
# SwinMSP 微调模块
# --------------------------------------------------------
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import argparse
import datetime
import os
import time
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.cuda.amp as amp
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy, AverageMeter
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange
from config import get_config
from data import build_loader
from logger import create_logger
from lr_scheduler import build_scheduler
from models import build_model
from optimizer import build_optimizer
from utils_simmim import load_checkpoint, load_pretrained, save_checkpoint, get_grad_norm, auto_resume_helper
from validate import test4metrics, show_results, save2file, get_cls_map2
from DATA_PATH import PU_LABELS, XZ_LABELS, LK_LABELS

lab = {'PaviaU':PU_LABELS,'WHU_Hi_LongKou':LK_LABELS,'Xuzhou':XZ_LABELS}
def parse_option():
    parser = argparse.ArgumentParser('SimMIM fine-tuning script', add_help=False)
    parser.add_argument('--cfg', type=str, metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--pretrained', type=str, help='path to pre-trained model')
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
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')

    # distributed training
    parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')

    parser.add_argument("--runs", type=int,default=1, help="number of runs.")
    args = parser.parse_args()

    config = get_config(args)

    return args, config


def main(config):
    dataloader_finetune, dataloader_val, dataloader_test, mixup_fn, weights, dataloader_all, gt = build_loader(config, swin_mae=True, is_pretrain=False)
    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    num_patches = dataloader_finetune.dataset.x_data.shape[3]

    model = build_model(config, is_pretrain=False, num_patches=num_patches)
    model.cuda()
    logger.info(str(model))

    optimizer = build_optimizer(config, model, simmim=True, is_pretrain=False)
    model_without_ddp = model

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    if hasattr(model_without_ddp, 'flops'):
        flops = model_without_ddp.flops()
        logger.info(f"number of GFLOPs: {flops / 1e9}")

    lr_scheduler = build_scheduler(config, optimizer, len(dataloader_finetune.dataset))
    scaler = amp.GradScaler()

    if config.AUG.MIXUP > 0.:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif config.MODEL.LABEL_SMOOTHING > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=config.MODEL.LABEL_SMOOTHING)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    # 开启损失权重
    if config.MODEL.LOSS_WEIGHTS:
        weights = torch.HalfTensor(weights)
        criterion = torch.nn.CrossEntropyLoss(weight=weights.cuda())

    max_accuracy = 0.0

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

    # 手动指定恢复路径
    if config.MODEL.RESUME:
        max_accuracy = load_checkpoint(config, model_without_ddp, optimizer, lr_scheduler, scaler, logger)
        acc1, acc5, loss = validate(config, dataloader_test, model)
        logger.info(f"Accuracy of the network on the {len(dataloader_test.dataset)} test images: {acc1:.3f}%")
        if config.EVAL_MODE:
            return

    file_time = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime()) + '_' + config.TAG
    # 加载预训练参数
    if config.MODEL.PRETRAINED and (not config.MODEL.RESUME):
        load_pretrained(config, model_without_ddp, logger)
        # acc1, acc5, loss = validate(config, data_loader_val, model)
        # logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.3f}%")

    if config.THROUGHPUT_MODE:
        throughput(dataloader_test, model, logger)
        return

    logger.info("Start training ..........")
    start_time = time.time()

    logpath = r'./log/swin_msp_ft/'
    if not os.path.exists(logpath):
        os.makedirs(logpath)
    writer = SummaryWriter(logpath + file_time + "_FINETUNE_" + config.TAG if config.TAG else '')

    # 将模型和参数写入每次训练日志
    writer.add_text("config", "<pre>" + config.dump() + "</pre>")
    writer.add_text("model", "<pre>" + str(model) + "</pre>")

    # random seeds
    seeds = [11111, 22222, 33333, 44444, 55555, 66666, 77777, 88888, 99999, 101010]
    # empty list to storing results
    results = []
    runs = config.TRAIN.RUNS
    for i in range(runs):
        np.random.seed(seeds[i])
        for epoch in trange(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS, postfix='Fine-tuning: '+config.TAG):

            train_one_epoch(config, model, criterion, dataloader_finetune, optimizer, epoch, mixup_fn, lr_scheduler, scaler,
                            writer)

            if epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1):
                save_checkpoint(config, epoch, model_without_ddp, max_accuracy, optimizer, lr_scheduler, scaler, logger)

            # 微调验证
        logger.info(f"Round {i} train finished.")
        logger.info("Start Testing ..........")
        run_results = test4metrics(model, dataloader_test)
        results.append(run_results)
        text = show_results(run_results, lab[config.DATA.DATASET][1:])
        save2file(i, text, config.DATA.DATASET, file_time, logger)

    if runs >= 1:
        text = show_results(results, lab[config.DATA.DATASET][1:], agregated=True)
        save2file(runs, text, config.DATA.DATASET, file_time, logger)
        get_cls_map2(runs, model, config.DATA.DATASET, dataloader_all, gt,file_time, logger)
        writer.add_text("cls_result", "<pre>" + text + "</pre>")

    writer.close()
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))


def train_one_epoch(config, model, criterion, data_loader, optimizer, epoch, mixup_fn, lr_scheduler, scaler, writer):
    model.train()
    optimizer.zero_grad()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()
    loss_scale_meter = AverageMeter()

    start = time.time()
    end = time.time()
    # for idx, (samples, targets) in tqdm(enumerate(data_loader)):
    for idx, (samples, targets) in enumerate(data_loader):
        samples = samples.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        outputs = model(samples)

        if config.TRAIN.ACCUMULATION_STEPS > 1:
            loss = criterion(outputs, targets)
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
            loss = criterion(outputs, targets)

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

        torch.cuda.synchronize()
        # 这个图像不具有参考性
        writer.add_scalar("finetune_loss", loss, epoch)

        loss_meter.update(loss.item(), targets.size(0))
        norm_meter.update(grad_norm)
        loss_scale_meter.update(scaler.get_scale())
        batch_time.update(time.time() - end)
        end = time.time()

        # if idx % config.PRINT_FREQ == 0:
        #     lr = optimizer.param_groups[-1]['lr']
        #     memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
        #     etas = batch_time.avg * (num_steps - idx)
        #     logger.info(
        #         f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
        #         f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t'
        #         f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
        #         f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
        #         f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
        #         f'loss_scale {loss_scale_meter.val:.4f} ({loss_scale_meter.avg:.4f})\t'
        #         f'mem {memory_used:.0f}MB')
    epoch_time = time.time() - start
    # logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")


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
        output = model(images)

        # measure accuracy and record loss
        loss = criterion(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        # acc1 = reduce_tensor(acc1)
        # acc5 = reduce_tensor(acc5)
        # loss = reduce_tensor(loss)

        loss_meter.update(loss.item(), target.size(0))
        acc1_meter.update(acc1.item(), target.size(0))
        acc5_meter.update(acc5.item(), target.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # if idx % config.PRINT_FREQ == 0:
        #     memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
        #     logger.info(
        #         f'Test: [{idx}/{len(data_loader)}]\t'
        #         f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
        #         f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
        #         f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
        #         # f'Acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f})\t'
        #         # f'Mem {memory_used:.0f}MB'
        #     )
    logger.info(f' * Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f}')
    return acc1_meter.avg, acc5_meter.avg, loss_meter.avg


@torch.no_grad()
def throughput(data_loader, model, logger):
    # summary(model, torch.zeros((1, 5, 5, 100), device='cuda:0'))
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
    _, config = parse_option()

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1
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
