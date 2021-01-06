
import time
import os, sys, math, argparse

import cv2
import numpy as np
import torch
import torch.nn as nn
from easydict import EasyDict as edict

from cfg import Cfg


@torch.no_grad()
def evaluate(model, data_loader, cfg, device, logger=None, **kwargs):
    """
    evaluator:
      model
      data_loader
      config
      device
      logger
      kwargs
    """


    for images, targets in data_loader:

        images = images.to(device=device, dtype=torch.float32)
        # targets = targets.to(device=device)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()

        outputs = model(images, inference=True)
        # outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time
        import pdb; pdb.set_trace()



def get_args(**kwargs):
    cfg = kwargs
    parser = argparse.ArgumentParser(
      description='Train the Model on images and target masks',
      formatter_class=argparse.ArgumentDefaultsHelpFormatter )
    parser.add_argument( '-b', '--batch-size', metavar='B', type=int,
      nargs='?', default=8, help='Batch size', dest='batchsize' )
    parser.add_argument( '-l', '--learning-rate', metavar='LR', type=float,
      nargs='?', default=0.001, help='Learning rate', dest='learning_rate' )
    parser.add_argument( '-f', '--load', dest='load', type=str, default=None,
      help='Load model from a .pth file' )
    parser.add_argument( '-g', '--gpu', metavar='G', type=str, default='-1',
      help='GPU', dest='gpu' )
    parser.add_argument( '-dir', '--data-dir', type=str, default=None,
      help='dataset dir', dest='dataset_dir' )
    parser.add_argument( '-pretrained', type=str, default=None,
      help='pretrained yolov4.conv.137' )
    parser.add_argument( '-classes', type=int, default=2,
      help='dataset classes' )
    parser.add_argument( '-train_label_path', dest='train_label', type=str,
      default='/home/iain/data/asus_video/artic_train.txt', help="training label path" )
    parser.add_argument( '-val_label_path', dest='val_label', type=str,
      default='/home/iain/data/asus_video/artic_valid.txt', help="validation label path" )
    parser.add_argument( '-optimizer', type=str, default='adam',
      help='training optimizer', dest='TRAIN_OPTIMIZER' )
    parser.add_argument( '-iou-type', type=str, default='iou',
      help='iou type (iou, giou, diou, ciou)', dest='iou_type')
    parser.add_argument( '-keep-checkpoint-max', type=int, default=10,
      help='maximum number of checkpoints to keep. If set 0, all checkpoints will be kept',
      dest='keep_checkpoint_max' )
    args = vars(parser.parse_args())

    # for k in args.keys():
    #     cfg[k] = args.get(k)
    cfg.update(args)

    return edict(cfg)


if __name__ == "__main__":

    from torch.utils.data import DataLoader
    from artic_dataset import Artic_dataset
    from artic_model import ArticYolo

    config = get_args(**Cfg)
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    val_dataset = Artic_dataset( config.val_label,
      config.width, config.height, config.maxboxes, train=False )
    val_loader = DataLoader(val_dataset,
      batch_size=config.batch // config.subdivisions, shuffle=True,
      num_workers=8, pin_memory=True, drop_last=True,
      collate_fn=val_dataset.collate
    )

    model = ArticYolo(
      config.pretrained, n_classes=config.classes, n_anchors=config.n_anchors )
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.to(device=device)
    model.eval()

    try:
        evaluate(model, val_loader, config, device)
    except KeyboardInterrupt:
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
