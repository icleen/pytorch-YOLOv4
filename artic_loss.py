
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
          [-91.9571145753918, -51.783354842452134, -46.069719753930265, 30.758714969241282, 128.84151526390733, 6.359553311472068, 99.66329966329967, 11.880711880711887],
          [0.5844155844155778, -69.54545454545458, 35.419126328217246, -68.71310507674147, -59.56413505656803, 94.72865577911871, 23.47528701643268, 63.33401615203975],
          [-104.68975468975474, 27.633477633477625, -123.84007174060798, 50.03133239029986, 64.48551448551449, 64.53546453546454, 80.67532467532466, 105.61038961038965],
          [-46.808551533263945, -11.10214172910984, -47.16359581569602, -74.62645161246026, 47.857225442979846, -13.947732702152413, -78.13852813852816, 122.29437229437231],
          [-73.37662337662339, -103.5714285714286, -157.34265734265733, 22.07792207792208, 121.1255411255411, 69.95670995670999, 46.21923684571134, 9.10980039657073],
          [-142.34740075946587, -8.170530984341921, -105.98555805607775, -18.597390308588295, 75.45454545454545, -63.1818181818182, 132.99512987012986, -5.844155844155841],
          [-107.69600769600773, -23.713323713323707, 14.912277213296845, -30.898443557184734, 12.65527238498558, 54.54996637738744, -18.441558441558442, 129.4805194805195],
          [-52.83996115441124, -62.40326281976513, -0.5038743278147152, -88.14292368088361, 98.15033451397083, -19.047619047619058, 138.17866027993273, 69.7428743471972],
          [-111.29870129870133, 69.6103896103896, -32.00916730328496, -25.668449197860962, 55.62770562770559, 134.41558441558445, 84.35064935064932, 52.98701298701299]
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

            # loss calculation
            output[..., self.n_preds] *= obj_mask
            output[..., np.r_[0:self.n_preds, self.n_conf:self.n_ch]] *= tgt_mask
            output[..., 2:self.n_preds] *= tgt_scale

            target[..., self.n_preds] *= obj_mask
            target[..., np.r_[0:self.n_preds, self.n_conf:self.n_ch]] *= tgt_mask
            target[..., 2:self.n_preds] *= tgt_scale

            loss_xy += F.binary_cross_entropy(
              input=output[..., :2], target=target[..., :2],
              weight=(tgt_scale * tgt_scale)[..., :2], reduction='sum'
            )
            loss_wh += F.mse_loss( input=output[..., 2:self.n_preds],
              target=target[..., 2:self.n_preds], reduction='sum' ) / (self.n_preds-2)
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
          self.n_preds + self.n_classes ).to(self.device)
        obj_mask = torch.ones( batchsize, self.n_anchors, nH, nW ).to(self.device)
        tgt_scale = torch.zeros( batchsize, self.n_anchors, nH, nW, self.n_preds-2).to(self.device)
        target = torch.zeros( batchsize, self.n_anchors, nH, nW, self.n_ch ).to(self.device)

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
            truth_box = torch.zeros(n, self.n_preds).to(self.device)
            truth_box[:n, 2:] = truth_labels[b, :n, 2:]
            truth_i = truth_i_all[b, :n]
            truth_j = truth_j_all[b, :n]

            # calculate iou between truth and reference anchors
            anchor_ious_all = cross_iou(
              truth_box.cpu(),
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
              truth_box,
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
                    target[b, a, j, i, self.n_conf + labels[b, ti, self.n_preds].to(torch.int16).cpu().numpy()] = 1
                    tgt_scale[b, a, j, i, :] = torch.sqrt(2 - truth_w_all[b, ti] * truth_h_all[b, ti] / nH / nW)
        return obj_mask, tgt_mask, tgt_scale, target
