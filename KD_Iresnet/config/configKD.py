from easydict import EasyDict as edict

cfg = edict()
cfg.dataset = "emore"  # MS1MV2
cfg.embedding_size = 512
cfg.momentum = 0.9
cfg.weight_decay = 5e-4
cfg.batch_size = 8
cfg.lr = 0.1
cfg.output = "output/FaceRecognitionKD"
cfg.scale=1.0
cfg.global_step=0
cfg.s=64.0
cfg.m=0.5


# for KD
cfg.teacher_pth = "/home/djordacevic/PyProj/KD_Iresnet/weights/r34/"
cfg.teacher_global_step = 295672
cfg.teacher_network="iresnet32"

# if use pretrained model (not for resume!)
cfg.student_pth = "/home/djordacevic/PyProj/KD_Iresnet/weights/r18"
cfg.student_global_step = 0
cfg.net_name="iresnet18"

cfg.w=100

cfg.rec = "/home/djordacevic/Datasets/faces_emore"
cfg.num_classes = 85742
cfg.num_image = 5822653
cfg.num_epoch = 1
cfg.warmup_epoch = -1
cfg.val_targets = ["lfw"]#, "agedb_30"]
cfg.eval_step = 5686

def lr_step_func(epoch):
    return ((epoch + 1) / (4 + 1)) ** 2 if epoch < -1 else 0.1 ** len(
        [m for m in [8, 14, 20, 25] if m - 1 <= epoch])

cfg.lr_func = lr_step_func
