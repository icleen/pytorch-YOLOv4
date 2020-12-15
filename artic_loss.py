
import time
from loss import *


def cross_iou(bboxes_a, bboxes_b, maxdist=418, GIoU=False, DIoU=False, CIoU=False):
    """Calculate the Intersection of Unions (IoUs) between bounding boxes.
    IoU is calculated as a ratio of area of the intersection
    and area of the union.

    Args:
        bbox_a (array): An array whose shape is :math:`(N, 10)`.
            :math:`N` is the number of bounding boxes.
            The dtype should be :obj:`numpy.float32`.
        bbox_b (array): An array similar to :obj:`bbox_a`,
            whose shape is :math:`(K, 10)`.
            The dtype should be :obj:`numpy.float32`.
    Returns:
        array:
        An array whose shape is :math:`(N, K)`. \
        An element at index :math:`(n, k)` contains IoUs between \
        :math:`n` th bounding box in :obj:`bbox_a` and :math:`k` th bounding \
        box in :obj:`bbox_b`.

    from: https://github.com/chainer/chainercv
    https://github.com/ultralytics/yolov3/blob/eca5b9c1d36e4f73bf2f94e141d864f1c2739e23/utils/utils.py#L262-L282
    """

    ious = torch.zeros((bboxes_a.shape[0], bboxes_b.shape[0]))
    for bi, bbox in enumerate(bboxes_a):
        ious[bi] = maxdist - torch.sqrt(torch.sum(torch.pow(bboxes_b - bbox, 2), 1))
    return ious


class Artic_loss(nn.Module):
    def __init__( self, n_classes=2, n_anchors=3,
      device=None, batch=2, image_size=(480, 480) ):
        super(Artic_loss, self).__init__()
        self.device = device
        self.strides = [8, 16, 32]
        self.n_classes = n_classes
        self.n_anchors = n_anchors
        self.n_preds = 10 # 4 for regular yolo
        self.n_conf = self.n_preds + 1
        self.n_ch = (self.n_preds + 1 + self.n_classes)
        self.height, self.width = image_size
        self.maxdist = np.sqrt(self.height**2 + self.width**2)

        # self.anchors = [
        #   [12, 16], [19, 36], [40, 28],
        #   [36, 75], [76, 55], [72, 146],
        #   [142, 110], [192, 243], [459, 401]
        # ]
        self.anchors = np.array([
          [32.251082251082266, -54.502164502164504, -109.0909090909091, -139.44805194805195, -45.65062739411482, -60.24401349749069, 87.75510204081634, 16.790352504638232],
          [-147.3214285714286, -127.2727272727273, 114.24047590585879, 19.613193846399632, 235.95779220779218, -8.40097402597403, 212.24911452184185, 73.34710743801655],
          [-105.47309833024121, 77.17996289424863, -209.39060939060943, 31.26873126873128, 72.72727272727276, 152.1645021645022, 14.18181818181818, -45.298701298701296],
          [104.67532467532469, -63.896103896103895, 74.79338842975207, -152.8925619834711, 58.05909835572687, -96.6583076677022, 213.73889268626115, -28.229665071770334],
          [-38.6068476977568, -51.18063754427394, 46.51274651274653, 29.822029822029823, 147.52066115702476, -97.99291617473438, 34.84848484848486, 170.34632034632037],
          [56.01731601731602, -157.5757575757576, -46.9155844155844, -97.72727272727273, 77.70562770562769, 36.14718614718617, -1.00851227011111e-14, 15.793883535819015],
          [-166.52236652236655, -57.21500721500721, 44.99618029029792, -44.07944996180291, -79.03581822700318, 5.874452688822191, -122.72727272727269, 164.935064935065],
          [6.161923183199777, 5.913235700469749, 132.6599326599327, -33.429533429533436, 174.6753246753247, 88.6936592818946, 146.01113172541747, -133.48794063079782],
          [-206.67903525046384, -2.5974025974026023, -23.37662337662337, 64.93506493506496, -89.61038961038962, 168.1818181818182, 140.72356215213355, 133.95176252319112]
        ], dtype=np.float32)
        self.anch_masks = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        self.ignore_thre = 0.5

        self.masked_anchors, self.ref_anchors, self.grid_x = [], [], []
        self.grid_y, self.anchor_w, self.anchor_h = [], [], []

        for i in range(len(self.strides)):
            all_anchors_grid = self.anchors / self.strides[i]
            # all_anchors_grid = np.array([[val / self.strides[i] for val in anch]
            #   for anch in self.anchors], dtype=np.float32)
            masked_anchors = np.array([all_anchors_grid[j]
              for j in self.anch_masks[i]], dtype=np.float32)
            ref_anchors = np.zeros((len(all_anchors_grid), self.n_preds), dtype=np.float32)
            ref_anchors[:, 2:] = all_anchors_grid
            ref_anchors = torch.from_numpy(ref_anchors)
            # calculate pred - xywh obj cls
            nW = self.width // self.strides[i]
            nH = self.height // self.strides[i]
            grid_x = ( torch.arange(nW, dtype=torch.float)
              .repeat( batch, n_anchors, nW, 1 ).to(device) )
            grid_y = ( torch.arange(nH, dtype=torch.float)
              .repeat( batch, n_anchors, nH, 1 ).permute(0, 1, 3, 2).to(device) )
            anchor_w = ( torch.from_numpy(masked_anchors[:, 0])
              .repeat( batch, nH, nW, 1 ).permute(0, 3, 1, 2)
              .to(device).unsqueeze(-1) )
            anchor_h = ( torch.from_numpy(masked_anchors[:, 1])
              .repeat( batch, nH, nW, 1 ).permute(0, 3, 1, 2)
              .to(device).unsqueeze(-1) )

            self.masked_anchors.append(masked_anchors)
            self.ref_anchors.append(ref_anchors)
            self.grid_x.append(grid_x)
            self.grid_y.append(grid_y)
            self.anchor_w.append(anchor_w)
            self.anchor_h.append(anchor_h)


    def forward(self, xin, labels=None):
        loss, loss_xy, loss_wh, loss_obj, loss_cls, loss_l2 = 0, 0, 0, 0, 0, 0
        for output_id, output in enumerate(xin):
            batchsize = output.shape[0]
            nH = output.shape[2]
            nW = output.shape[3]

            output = output.view(batchsize, self.n_anchors, self.n_ch, nH, nW)
            output = output.permute(0, 1, 3, 4, 2)  # .contiguous()

            """
            output shape = ( batchsize, anchors, nH, nW,
              (preds + confidence + classes) )
            """
            # logistic activation for xy, obj, cls
            output[..., np.r_[:2, self.n_preds:self.n_ch]] = torch.sigmoid(
              output[..., np.r_[:2, self.n_preds:self.n_ch]] )

            pred = output[..., :self.n_preds].clone()
            pred[..., 0] += self.grid_x[output_id]
            pred[..., 1] += self.grid_y[output_id]
            pred[..., 2::2] = torch.exp(pred[..., 2::2]) * self.anchor_w[output_id]
            pred[..., 3::2] = torch.exp(pred[..., 3::2]) * self.anchor_h[output_id]

            obj_mask, tgt_mask, tgt_scale, target = self.build_target(
              pred, labels, batchsize, nH, nW, output_id )

            import pdb; pdb.set_trace()
            # loss calculation
            output[..., self.n_preds] *= obj_mask
            output[..., np.r_[0:self.n_preds, self.n_conf:self.n_ch]] *= tgt_mask
            output[..., 2:self.n_preds] *= tgt_scale

            target[..., self.n_preds] *= obj_mask
            target[..., np.r_[0:self.n_preds, self.n_conf:self.n_ch]] *= tgt_mask
            target[..., 2:self.n_preds] *= tgt_scale

            loss_xy += F.binary_cross_entropy(
              input=output[..., :2], target=target[..., :2],
              weight=tgt_scale * tgt_scale, reduction='sum'
            )
            loss_wh += F.mse_loss( input=output[..., 2:self.n_preds],
              target=target[..., 2:self.n_preds], reduction='sum' ) / 2
            loss_obj += F.binary_cross_entropy( input=output[..., self.n_preds],
              target=target[..., self.n_preds], reduction='sum' )
            loss_cls += F.binary_cross_entropy( input=output[..., self.n_conf:],
              target=target[..., self.n_conf:], reduction='sum' )
            loss_l2 += F.mse_loss(input=output, target=target, reduction='sum')

        loss = loss_xy + loss_wh + loss_obj + loss_cls

        return loss, loss_xy, loss_wh, loss_obj, loss_cls, loss_l2

    def build_target(self, pred, labels, batchsize, nH, nW, output_id):
        # target assignment
        tgt_mask = torch.zeros( batchsize, self.n_anchors, nH, nW,
          self.n_preds + self.n_classes )
        obj_mask = torch.ones( batchsize, self.n_anchors, nH, nW )
        tgt_scale = torch.zeros( batchsize, self.n_anchors, nH, nW, 2)
        target = torch.zeros( batchsize, self.n_anchors, nH, nW, self.n_ch )

        # labels = labels.cpu().data
        nlabel = (labels.sum(dim=2) > 0).sum(dim=1)  # number of objects
        truth_labels = labels[..., 0:-1] / self.strides[output_id]
        truth_w_all = torch.zeros( labels.shape[0], labels.shape[1], 1 )
        truth_h_all = torch.zeros( labels.shape[0], labels.shape[1], 1 )
        truth_w_all[..., 0] = ( labels[..., 2:-1:2].max(-1)[0] - labels[..., 2:-1:2].min(-1)[0] ) /self.strides[output_id]
        truth_h_all[..., 0] = ( labels[..., 3:-1:2].max(-1)[0] - labels[..., 3:-1:2].min(-1)[0] ) /self.strides[output_id]
        truth_i_all = truth_labels[..., 0].to(torch.int16).cpu().numpy()
        truth_j_all = truth_labels[..., 1].to(torch.int16).cpu().numpy()

        for b in range(batchsize):
            n = int(nlabel[b])
            if n == 0:
                continue
            truth_box = torch.zeros(n, self.n_preds)
            truth_box[:n, 2:] = truth_labels[b, :n, 2:]
            truth_i = truth_i_all[b, :n]
            truth_j = truth_j_all[b, :n]

            # calculate iou between truth and reference anchors
            anchor_ious_all = cross_iou(
              truth_box,
              self.ref_anchors[output_id],
              maxdist=self.maxdist/self.strides[output_id] )

            best_n_all = anchor_ious_all.argmax(dim=1)
            best_n = best_n_all % 3
            best_n_mask = ( (best_n_all == self.anch_masks[output_id][0]) |
                            (best_n_all == self.anch_masks[output_id][1]) |
                            (best_n_all == self.anch_masks[output_id][2]) )

            if sum(best_n_mask) == 0:
                continue

            truth_box[:n, :2] = truth_labels[b, :n, :2]

            pred_ious = cross_iou(
              pred[b].view(-1, self.n_preds),
              truth_box.to(self.device),
              maxdist=self.maxdist/self.strides[output_id] )

            pred_best_iou, _ = pred_ious.max(dim=1)
            pred_best_iou = (pred_best_iou > self.ignore_thre)
            pred_best_iou = pred_best_iou.view(pred[b].shape[:3])
            # set mask to zero (ignore) if pred matches truth
            obj_mask[b] = ~ pred_best_iou

            for ti in range(best_n.shape[0]):
                if best_n_mask[ti] == 1:
                    i, j = truth_i[ti], truth_j[ti]
                    a = best_n[ti]
                    obj_mask[b, a, j, i] = 1
                    tgt_mask[b, a, j, i, :] = 1
                    target[b, a, j, i, 0] = truth_labels[b, ti, 0] - truth_labels[b, ti, 0].to(torch.int16).to(torch.float)
                    target[b, a, j, i, 1] = truth_labels[b, ti, 1] - truth_labels[b, ti, 1].to(torch.int16).to(torch.float)
                    target[b, a, j, i, 2:self.n_preds] = torch.log(
                        truth_labels[b, ti, 2:] / torch.Tensor(self.masked_anchors[output_id])[best_n[ti], 0] + 1e-16
                    )
                    target[b, a, j, i, self.n_preds] = 1
                    target[b, a, j, i, self.n_conf + labels[b, ti, 4].to(torch.int16).cpu().numpy()] = 1
                    tgt_scale[b, a, j, i, :] = torch.sqrt(2 - truth_w_all[b, ti] * truth_h_all[b, ti] / nH / nW)
        return obj_mask, tgt_mask, tgt_scale, target
