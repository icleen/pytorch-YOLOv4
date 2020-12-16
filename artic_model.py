import torch
from torch import nn
import torch.nn.functional as F
from tool.torch_utils import *
from models import *
from tool.artic_layer import ArticLayer

class DownSample5(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv_Bn_Activation(512, 256, 3, 2, 'mish')
        self.conv2 = Conv_Bn_Activation(256, 512, 1, 1, 'mish')
        self.conv3 = Conv_Bn_Activation(256, 256, 1, 1, 'mish')

        self.resblock = ResBlock(ch=256, nblocks=4)
        self.conv4 = Conv_Bn_Activation(256, 512, 1, 1, 'mish')
        self.conv5 = Conv_Bn_Activation(1024, 512, 1, 1, 'mish')

    def forward(self, input):
        x1 = self.conv1(input)
        x2 = self.conv2(x1)
        x3 = self.conv3(x1)

        r = self.resblock(x3)
        x4 = self.conv4(r)

        x4 = torch.cat([x4, x2], dim=1)
        x5 = self.conv5(x4)
        return x5

class ArticNeck(nn.Module):
    def __init__(self, inference=False):
        super().__init__()
        self.inference = inference

        self.conv1 = Conv_Bn_Activation(512, 256, 1, 1, 'leaky')
        self.conv2 = Conv_Bn_Activation(256, 512, 3, 1, 'leaky')
        self.conv3 = Conv_Bn_Activation(512, 256, 1, 1, 'leaky')
        # SPP
        self.maxpool1 = nn.MaxPool2d(kernel_size=5, stride=1, padding=5 // 2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=9, stride=1, padding=9 // 2)
        self.maxpool3 = nn.MaxPool2d(kernel_size=13, stride=1, padding=13 // 2)

        # R -1 -3 -5 -6
        # SPP
        self.conv4 = Conv_Bn_Activation(1024, 512, 1, 1, 'leaky')
        # self.conv5 = Conv_Bn_Activation(512, 1024, 3, 1, 'leaky')
        # self.conv6 = Conv_Bn_Activation(1024, 512, 1, 1, 'leaky')
        self.conv5 = Conv_Bn_Activation(512, 512, 3, 1, 'leaky')
        self.conv7 = Conv_Bn_Activation(512, 256, 1, 1, 'leaky')
        # UP
        self.upsample1 = Upsample()
        # R 85
        self.conv8 = Conv_Bn_Activation(512, 256, 1, 1, 'leaky')
        # R -1 -3
        self.conv9 = Conv_Bn_Activation(512, 256, 1, 1, 'leaky')
        self.conv10 = Conv_Bn_Activation(256, 512, 3, 1, 'leaky')
        # self.conv11 = Conv_Bn_Activation(512, 256, 1, 1, 'leaky')
        # self.conv12 = Conv_Bn_Activation(256, 512, 3, 1, 'leaky')
        self.conv13 = Conv_Bn_Activation(512, 256, 1, 1, 'leaky')
        self.conv14 = Conv_Bn_Activation(256, 128, 1, 1, 'leaky')
        # UP
        self.upsample2 = Upsample()
        # R 54
        self.conv15 = Conv_Bn_Activation(256, 128, 1, 1, 'leaky')
        # R -1 -3
        self.conv16 = Conv_Bn_Activation(256, 128, 1, 1, 'leaky')
        # self.conv17 = Conv_Bn_Activation(128, 256, 3, 1, 'leaky')
        # self.conv18 = Conv_Bn_Activation(256, 128, 1, 1, 'leaky')
        self.conv19 = Conv_Bn_Activation(128, 256, 3, 1, 'leaky')
        self.conv20 = Conv_Bn_Activation(256, 128, 1, 1, 'leaky')

    def forward(self, input, downsample4, downsample3, inference=False):
        x1 = self.conv1(input)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)

        # SPP
        m1 = self.maxpool1(x3)
        m2 = self.maxpool2(x3)
        m3 = self.maxpool3(x3)
        spp = torch.cat([m3, m2, m1, x3], dim=1)
        # SPP end
        x4 = self.conv4(spp)
        x5 = self.conv5(x4)
        # x6 = self.conv6(x5)
        # x7 = self.conv7(x6)
        x7 = self.conv7(x5)
        # UP
        up = self.upsample1(x7, downsample4.size(), self.inference)
        # R 85
        x8 = self.conv8(downsample4)
        # R -1 -3
        x8 = torch.cat([x8, up], dim=1)

        x9 = self.conv9(x8)
        x10 = self.conv10(x9)
        # x11 = self.conv11(x10)
        # x12 = self.conv12(x11)
        # x13 = self.conv13(x12)
        x13 = self.conv13(x10)
        x14 = self.conv14(x13)

        # UP
        up = self.upsample2(x14, downsample3.size(), self.inference)
        # R 54
        x15 = self.conv15(downsample3)
        # R -1 -3
        x15 = torch.cat([x15, up], dim=1)

        x16 = self.conv16(x15)
        # x17 = self.conv17(x16)
        # x18 = self.conv18(x17)
        # x19 = self.conv19(x18)
        x19 = self.conv19(x16)
        x20 = self.conv20(x19)
        return x20, x13, x5

class ArticHead(nn.Module):
    def __init__(self, output_ch, n_classes, inference=False):
        super().__init__()
        self.inference = inference

        self.conv1 = Conv_Bn_Activation(128, 256, 3, 1, 'leaky')
        self.conv2 = Conv_Bn_Activation(256, output_ch, 1, 1, 'linear', bn=False, bias=True)

        self.yolo1 = ArticLayer(
          anchor_mask=[0, 1, 2], num_classes=n_classes,
          anchors=[ 12, 16, 19, 36, 40, 28, 36, 75, 76,
            55, 72, 146, 142, 110, 192, 243, 459, 401 ],
          num_anchors=9, stride=8
        )

        # R -4
        self.conv3 = Conv_Bn_Activation(128, 256, 3, 2, 'leaky')

        # R -1 -16
        self.conv4 = Conv_Bn_Activation(512, 256, 1, 1, 'leaky')
        self.conv5 = Conv_Bn_Activation(256, 512, 3, 1, 'leaky')
        # self.conv6 = Conv_Bn_Activation(512, 256, 1, 1, 'leaky')
        # self.conv7 = Conv_Bn_Activation(256, 512, 3, 1, 'leaky')
        self.conv8 = Conv_Bn_Activation(512, 256, 1, 1, 'leaky')
        self.conv9 = Conv_Bn_Activation(256, 512, 3, 1, 'leaky')
        self.conv10 = Conv_Bn_Activation(512, output_ch, 1, 1, 'linear', bn=False, bias=True)

        self.yolo2 = ArticLayer(
          anchor_mask=[3, 4, 5], num_classes=n_classes,
          anchors=[ 12, 16, 19, 36, 40, 28, 36, 75, 76,
          55, 72, 146, 142, 110, 192, 243, 459, 401 ],
          num_anchors=9, stride=16
        )

        # R -4
        self.conv11 = Conv_Bn_Activation(256, 512, 3, 2, 'leaky')

        # R -1 -37
        self.conv12 = Conv_Bn_Activation(1024, 512, 1, 1, 'leaky')
        self.conv13 = Conv_Bn_Activation(512, 512, 3, 1, 'leaky')
        # self.conv14 = Conv_Bn_Activation(512, 256, 1, 1, 'leaky')
        # self.conv15 = Conv_Bn_Activation(256, 512, 3, 1, 'leaky')
        self.conv16 = Conv_Bn_Activation(512, 256, 1, 1, 'leaky')
        self.conv17 = Conv_Bn_Activation(256, 512, 3, 1, 'leaky')
        self.conv18 = Conv_Bn_Activation(512, output_ch, 1, 1, 'linear', bn=False, bias=True)

        self.yolo3 = ArticLayer(
          anchor_mask=[6, 7, 8], num_classes=n_classes,
          anchors=[ 12, 16, 19, 36, 40, 28, 36, 75, 76,
            55, 72, 146, 142, 110, 192, 243, 459, 401 ],
          num_anchors=9, stride=32
        )

    def forward(self, input1, input2, input3):
        x1 = self.conv1(input1)
        x2 = self.conv2(x1)

        x3 = self.conv3(input1)
        # R -1 -16
        x3 = torch.cat([x3, input2], dim=1)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        # x6 = self.conv6(x5)
        # x7 = self.conv7(x6)
        # x8 = self.conv8(x7)
        x8 = self.conv8(x5)
        x9 = self.conv9(x8)
        x10 = self.conv10(x9)

        # R -4
        x11 = self.conv11(x8)
        # R -1 -37
        x11 = torch.cat([x11, input3], dim=1)

        x12 = self.conv12(x11)
        x13 = self.conv13(x12)
        # x14 = self.conv14(x13)
        # x15 = self.conv15(x14)
        # x16 = self.conv16(x15)
        x16 = self.conv16(x13)
        x17 = self.conv17(x16)
        x18 = self.conv18(x17)

        if self.inference:
            y1 = self.yolo1(x2)
            y2 = self.yolo2(x10)
            y3 = self.yolo3(x18)

            return get_region_boxes([y1, y2, y3])

        else:
            return [x2, x10, x18]

class ArticRegress(nn.Module):
    def __init__(self, output_ch, n_classes, inference=False):
        super().__init__()
        self.inference = inference

        self.conv1 = Conv_Bn_Activation(128, 256, 3, 1, 'leaky')
        self.conv2 = Conv_Bn_Activation(256, 128, 1, 1, 'linear', bn=False, bias=True)
        self.conv3 = Conv_Bn_Activation(128, 256, 3, 2, 'leaky')

        self.conv4 = Conv_Bn_Activation(256, 256, 1, 1, 'leaky')
        self.conv5 = Conv_Bn_Activation(256, 512, 3, 1, 'leaky')
        self.conv6 = Conv_Bn_Activation(512, 256, 1, 1, 'leaky')
        self.conv7 = Conv_Bn_Activation(256, 512, 3, 2, 'leaky')

        self.conv8 = Conv_Bn_Activation(512, 512, 1, 1, 'leaky')
        self.conv9 = Conv_Bn_Activation(512, 1024, 3, 1, 'leaky')
        self.conv10 = Conv_Bn_Activation(1024, 512, 1, 1, 'leaky')
        self.conv11 = Conv_Bn_Activation(512, 1024, 3, 1, 'leaky')
        self.conv12 = Conv_Bn_Activation(1024, output_ch, 1, 1, 'linear', bn=False, bias=True)

        self.regress = RegressLayer(n_classes)


    def forward(self, input, _, _):
        x1 = self.conv1(input)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        x7 = self.conv7(x6)
        x8 = self.conv8(x7)
        x9 = self.conv9(x8)
        x0 = self.conv10(x9)
        x11 = self.conv11(x10)
        x12 = self.conv12(x11)

        if self.inference:
            reg = self.regress(x2)

            return get_region_boxes(reg)

        else:
            return x12


class ArticYolo(nn.Module):
    def __init__(self, yolov4conv137weight=None, n_classes=2, inference=False, flatregress=True):
        super().__init__()

        # the number of predictions necessary + 1 for confidence prediction + the number of classes
        preds = 10 # 4 for regular yolo
        output_ch = (preds + 1 + n_classes) * 3

        # backbone
        self.down1 = DownSample1()
        self.down2 = DownSample2()
        self.down3 = DownSample3()
        self.down4 = DownSample4()
        self.down5 = DownSample5()
        # neck
        self.neck = ArticNeck(inference)
        # yolov4conv137
        if yolov4conv137weight:
            _model = nn.Sequential(
              self.down1, self.down2, self.down3,
              self.down4, self.down5, self.neck
            )
            pretrained_dict = torch.load(yolov4conv137weight)

            model_dict = _model.state_dict()
            # 1. filter out unnecessary keys
            pretrained_dict = {
              k1: v for (k, v), k1 in zip( pretrained_dict.items(), model_dict )
            }
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            _model.load_state_dict(model_dict)

        # head
        if flatregress:
            self.head = ArticRegress(output_ch, n_classes, inference)
        else:
            self.head = ArticHead(output_ch, n_classes, inference)


    def forward(self, input):
        d1 = self.down1(input)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)

        x20, x13, x6 = self.neck(d5, d4, d3)

        output = self.head(x20, x13, x6)
        return output
