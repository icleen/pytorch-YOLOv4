# -*- coding: utf-8 -*-
'''
@Time          : 2020/05/06 21:05
@Author        : Tianxiaomo
@File          : Cfg.py
@Noice         :
@Modificattion :
    @Author    :
    @Time      :
    @Detail    :

'''
import os
from easydict import EasyDict


_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

Cfg = EasyDict()

Cfg.use_darknet_cfg = True
Cfg.cfgfile = os.path.join(_BASE_DIR, 'cfg', 'yolov4.cfg')

Cfg.batch = 8
Cfg.subdivisions = 4
Cfg.width = 480
Cfg.height = 480
Cfg.channels = 3
Cfg.n_anchors = 1
Cfg.momentum = 0.949
Cfg.decay = 0.0005
Cfg.angle = 0
Cfg.saturation = 1.5
Cfg.exposure = 1.5
Cfg.hue = .1

Cfg.learning_rate = 0.00261
Cfg.burn_in = 1000
Cfg.max_batches = 500500
Cfg.steps = [400000, 450000]
Cfg.policy = Cfg.steps
Cfg.scales = .1, .1

Cfg.cutmix = 0
Cfg.mosaic = 1

Cfg.letter_box = 0
Cfg.jitter = 0.2
Cfg.classes = 2
Cfg.track = 0
Cfg.w = Cfg.width
Cfg.h = Cfg.height
Cfg.flip = 1
Cfg.blur = 0
Cfg.gaussian = 0
Cfg.boxes = 60  # box num
Cfg.TRAIN_EPOCHS = 300
Cfg.train_label = os.path.join(_BASE_DIR, 'data', 'train.txt')
Cfg.val_label = os.path.join(_BASE_DIR, 'data' ,'val.txt')
Cfg.TRAIN_OPTIMIZER = 'adam'
'''
image_path1 id,xc,yc,x1,y1,x2,y2,x3,y3,x4,y4 id,xc,yc,x1,y1,x2,y2,x3,y3,x4,y4 ...
image_path2 id,xc,yc,x1,y1,x2,y2,x3,y3,x4,y4 id,xc,yc,x1,y1,x2,y2,x3,y3,x4,y4 ...
...
'''

if Cfg.mosaic and Cfg.cutmix:
    Cfg.mixup = 4
elif Cfg.cutmix:
    Cfg.mixup = 2
elif Cfg.mosaic:
    Cfg.mixup = 3

Cfg.checkpoints = os.path.join(_BASE_DIR, 'checkpoints')
Cfg.TRAIN_TENSORBOARD_DIR = os.path.join(_BASE_DIR, 'log')

Cfg.iou_type = 'iou'  # 'giou', 'diou', 'ciou'

Cfg.keep_checkpoint_max = 10
