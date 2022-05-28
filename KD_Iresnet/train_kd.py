import argparse
import logging
import os
import time
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel.distributed import DistributedDataParallel
import torch.utils.data.distributed
from torch.nn.utils import clip_grad_norm_
from torch.nn import CrossEntropyLoss, MSELoss

from utils import losses
from config import configKD
from utils.dataset import MXFaceDataset, DataLoaderX
from utils.utils_callbacks import CallBackVerification, CallBackLoggingKD, CallBackModelCheckpointKD
from utils.utils_logging import AverageMeter, init_logging

from backbones.iresnet import iresnet18, iresnet34

torch.backends.cudnn.benchmark = True

def main(args):
    dist.init_process_group(rank=args.local_rank, backend='nccl', init_method='env://', world_size=1)
    local_rank = args.local_rank
    torch.cuda.set_device(local_rank)
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if not os.path.exists(configKD.cfg.output) and rank == 0:
        os.makedirs(configKD.cfg.output)
    else:
        time.sleep(2)

    log_root = logging.getLogger()
    init_logging(log_root, rank, configKD.cfg.output)
    trainset = MXFaceDataset(root_dir=configKD.cfg.rec, local_rank=local_rank)

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        trainset, shuffle=True)

    train_loader = DataLoaderX(
        local_rank=local_rank, dataset=trainset, batch_size=configKD.cfg.batch_size,
        sampler=train_sampler, num_workers=4, pin_memory=True, drop_last=True)

    # load teacher model
    backbone_teacher = iresnet34(num_features=configKD.cfg.embedding_size).to(local_rank)
    try:
        backbone_teacher_pth = os.path.join(configKD.cfg.teacher_pth, "backbone.pth")
        backbone_teacher.load_state_dict(torch.load(backbone_teacher_pth, map_location=torch.device(local_rank)))

        if rank == 0:
            logging.info("backbone teacher loaded successfully!")
    except (FileNotFoundError, KeyError, IndexError, RuntimeError):
        logging.info("load teacher backbone init, failed!")

    # load student model
    backbone_student = iresnet18(num_features=configKD.cfg.embedding_size).to(local_rank)

    if args.pretrained_student:
        try:
            backbone_student_pth = os.path.join(configKD.cfg.student_pth, "backbone.pth")
            backbone_student.load_state_dict(torch.load(backbone_student_pth, map_location=torch.device(local_rank)))

            if rank == 0:
                logging.info("backbone student loaded successfully!")
        except (FileNotFoundError, KeyError, IndexError, RuntimeError):
            logging.info("load student backbone init, failed!")

    if args.resume:
        try:
            backbone_student_pth = os.path.join(configKD.cfg.output, str(configKD.cfg.global_step) + "backbone.pth")
            backbone_student.load_state_dict(torch.load(backbone_student_pth, map_location=torch.device(local_rank)))

            if rank == 0:
                logging.info("backbone student resume loaded successfully!")
        except (FileNotFoundError, KeyError, IndexError, RuntimeError):
            logging.info("load student backbone resume init, failed!")

    #for ps in backbone_teacher.parameters():
    #    dist.broadcast(ps, 0)

    for ps in backbone_student.parameters():
        dist.broadcast(ps, 0)

    backbone_teacher = DistributedDataParallel(
        module=backbone_teacher, broadcast_buffers=False, device_ids=[local_rank])
    backbone_teacher.eval()

    backbone_student = DistributedDataParallel(
        module=backbone_student, broadcast_buffers=False, device_ids=[local_rank])
    backbone_student.train()

    # get header
    header = losses.ArcFace(in_features=configKD.cfg.embedding_size, out_features=configKD.cfg.num_classes, s=configKD.cfg.s, m=configKD.cfg.m).to(local_rank)
   
    if args.resume:
        try:
            header_pth = os.path.join(configKD.cfg.output, str(configKD.cfg.global_step) + "header.pth")
            header.load_state_dict(torch.load(header_pth, map_location=torch.device(local_rank)))

            if rank == 0:
                logging.info("header resume loaded successfully!")
        except (FileNotFoundError, KeyError, IndexError, RuntimeError):
            logging.info("header resume init, failed!")
    
    header = DistributedDataParallel(
        module=header, broadcast_buffers=False, device_ids=[local_rank])
    header.train()

    opt_backbone_student = torch.optim.SGD(
        params=[{'params': backbone_student.parameters()}],
        lr=configKD.cfg.lr / 512 * configKD.cfg.batch_size * world_size,
        momentum=0.9, weight_decay=configKD.cfg.weight_decay)
    opt_header = torch.optim.SGD(
        params=[{'params': header.parameters()}],
        lr=configKD.cfg.lr / 512 * configKD.cfg.batch_size * world_size,
        momentum=0.9, weight_decay=configKD.cfg.weight_decay)

    scheduler_backbone_student = torch.optim.lr_scheduler.LambdaLR(
        optimizer=opt_backbone_student, lr_lambda=configKD.cfg.lr_func)
    scheduler_header = torch.optim.lr_scheduler.LambdaLR(
        optimizer=opt_header, lr_lambda=configKD.cfg.lr_func)        

    criterion = CrossEntropyLoss()
    
    criterion2 = MSELoss()

    start_epoch = 0
    total_step = int(len(trainset) / configKD.cfg.batch_size / world_size * configKD.cfg.num_epoch)
    if rank == 0: logging.info("Total Step is: %d" % total_step)

    if args.resume:
        rem_steps = (total_step - configKD.cfg.global_step)
        cur_epoch = configKD.cfg.num_epoch - int(configKD.cfg.num_epoch / total_step * rem_steps)
        logging.info("resume from estimated epoch {}".format(cur_epoch))
        logging.info("remaining steps {}".format(rem_steps))
        
        start_epoch = cur_epoch
        scheduler_backbone_student.last_epoch = cur_epoch
        scheduler_header.last_epoch = cur_epoch

        # --------- this could be solved more elegant ----------------
        opt_backbone_student.param_groups[0]['lr'] = scheduler_backbone_student.get_lr()[0]
        opt_header.param_groups[0]['lr'] = scheduler_header.get_lr()[0]

        print("last learning rate: {}".format(scheduler_header.get_lr()))
        # ------------------------------------------------------------

    callback_verification = CallBackVerification(configKD.cfg.eval_step, rank, configKD.cfg.val_targets, configKD.cfg.rec) # 2000
    callback_logging = CallBackLoggingKD(50, rank, total_step, configKD.cfg.batch_size, world_size, writer=None)
    callback_checkpoint = CallBackModelCheckpointKD(rank, configKD.cfg.output)

    loss = AverageMeter()
    loss1 = AverageMeter()
    loss2 = AverageMeter()
    global_step = configKD.cfg.global_step

    for epoch in range(start_epoch, configKD.cfg.num_epoch):
        print(f'Epoch: {epoch}')
        train_sampler.set_epoch(epoch)
        for step, (masked_img, img, label) in enumerate(train_loader):
            global_step += 1
            img = img.cuda(local_rank, non_blocking=True)
            masked_img = masked_img.cuda(local_rank, non_blocking=True)

            label = label.cuda(local_rank, non_blocking=True)

            with torch.no_grad():
                features_teacher = F.normalize(backbone_teacher(img))
            features_student = F.normalize(backbone_student(masked_img))

            thetas = header(features_student, label)
            loss_v1 = criterion(thetas, label)
            loss_v2 = configKD.cfg.w*criterion2(features_student, features_teacher)
            loss_v = loss_v1 + loss_v2
            loss_v.backward()

            clip_grad_norm_(backbone_student.parameters(), max_norm=5, norm_type=2)

            opt_backbone_student.step()
            opt_header.step()

            opt_backbone_student.zero_grad()
            opt_header.zero_grad()

            loss.update(loss_v.item(), 1)
            loss1.update(loss_v1.item(), 1)
            loss2.update(loss_v2.item(), 1)
            
            callback_logging(global_step, loss, loss1, loss2, epoch)
            callback_verification(global_step, backbone_student)

        scheduler_backbone_student.step()
        scheduler_header.step()

        callback_checkpoint(global_step, backbone_student, header)

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training with template knowledge distillation')
    parser.add_argument('--local_rank', type=int, default=0, help='local_rank')
    parser.add_argument('--network_student', type=str, default="iresnet18", help="backbone of student network")
    parser.add_argument('--network_teacher', type=str, default="iresnet34", help="backbone of teacher network")
    parser.add_argument('--loss', type=str, default="ArcFace", help="loss function")
    parser.add_argument('--pretrained_student', type=int, default=1, help="use pretrained student model for KD")
    parser.add_argument('--resume', type=int, default=0, help="resume training")
    args_ = parser.parse_args()
    main(args_)
