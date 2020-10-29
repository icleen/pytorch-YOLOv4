
import time
from loss import *


class ArticRegionLoss(nn.Module):
    # n_classes=2, n_anchors=3,
    #   device=None, batch=2, image_size=480
    def __init__(self, n_classes=2, n_anchors=3):
        super(ArticRegionLoss, self).__init__()
        self.n_classes = n_classes
        self.anchors = [[12, 16], [19, 36], [40, 28], [36, 75], [76, 55], [72, 146], [142, 110], [192, 243], [459, 401]]
        self.n_anchors = n_anchors
        self.n_preds = 10 # 4 for regular yolo
        self.n_conf = self.n_preds + 1
        self.n_ch = (self.n_preds + 1 + self.n_classes)
        self.anchor_step = len(self.anchors) / self.n_anchors

        self.coord_scale = 1
        self.noobject_scale = 1
        self.object_scale = 5
        self.class_scale = 1
        self.thresh = 0.6
        self.seen = 0

        self.masked_anchors, self.ref_anchors, self.grid_x, self.grid_y, self.anchor_w, self.anchor_h = [], [], [], [], [], []

        for i in range(3):
            all_anchors_grid = [(w / self.strides[i], h / self.strides[i]) for w, h in self.anchors]
            masked_anchors = np.array([all_anchors_grid[j] for j in self.anch_masks[i]], dtype=np.float32)
            ref_anchors = np.zeros((len(all_anchors_grid), 4), dtype=np.float32)
            ref_anchors[:, 2:] = np.array(all_anchors_grid, dtype=np.float32)
            ref_anchors = torch.from_numpy(ref_anchors)
            # calculate pred - xywh obj cls
            fsize = image_size // self.strides[i]
            grid_x = torch.arange(nW, dtype=torch.float).repeat(
              batch, 3, nH, 1 ).to(device)
            grid_y = torch.arange(nH, dtype=torch.float).repeat(
              batch, 3, nW, 1 ).permute(0, 1, 3, 2).to(device)
            anchor_w = torch.from_numpy(masked_anchors[:, 0]).repeat(
              batch, nH, nW, 1 ).permute(0, 3, 1, 2).to(device).unsqueeze(-1)
            anchor_h = torch.from_numpy(masked_anchors[:, 1]).repeat(
              batch, nH, nW, 1 ).permute(0, 3, 1, 2).to(device).unsqueeze(-1)

            self.masked_anchors.append(masked_anchors)
            self.ref_anchors.append(ref_anchors)
            self.grid_x.append(grid_x)
            self.grid_y.append(grid_y)
            self.anchor_w.append(anchor_w)
            self.anchor_h.append(anchor_h)

    def forward(self, xin, target):
        # output : BxAs*n_ch*H*W
        # where n_ch = (n_preds+1+n_classes) (10+1+2)
        loss, loss_xy, loss_wh, loss_obj, loss_cls, loss_l2 = 0, 0, 0, 0, 0, 0
        for output_id, output in enumerate(xin):
            batchsize = output.shape[0]
            nH = output.shape[2]
            nW = output.shape[3]

            output = output.view(batchsize, self.n_anchors, self.n_ch, nH, nW)
            output = output.permute(0, 1, 3, 4, 2)  # .contiguous()
            # output : Batch*Anchors*H*W*(preds + confidence + classes)

            # sigmoid the x,y confidences classes
            output[..., np.r_[:2, self.n_preds:self.n_ch]] = torch.sigmoid(
              output[..., np.r_[:2, self.n_preds:self.n_ch]] )

            pred = output[..., :self.n_preds].clone()
            pred[..., 0] += self.grid_x[output_id]
            pred[..., 1] += self.grid_y[output_id]
            pred[..., 2::2] = torch.exp(pred[..., 2::2]) * self.anchor_w[output_id]
            pred[..., 3::2] = torch.exp(pred[..., 3::2]) * self.anchor_h[output_id]

            import pdb; pdb.set_trace()

            targets = build_targets( pred_boxes, target.data, nH, nW )
            nGT, nCorrect, coord_mask, conf_mask, cls_mask, tx, ty, tw, th, tconf, tcls = targets
            cls_mask = (cls_mask == 1)
            nProposals = int((conf > 0.25).sum().data[0])

            tx = Variable(tx.cuda())
            ty = Variable(ty.cuda())
            tw = Variable(tw.cuda())
            th = Variable(th.cuda())
            tconf = Variable(tconf.cuda())
            tcls = Variable(tcls.view(-1)[cls_mask].long().cuda())

            coord_mask = Variable(coord_mask.cuda())
            conf_mask = Variable(conf_mask.cuda().sqrt())
            cls_mask = Variable(cls_mask.view(-1, 1).repeat(1, nC).cuda())
            cls = cls[cls_mask].view(-1, nC)

            t3 = time.time()

            loss_x = self.coord_scale * nn.MSELoss(reduction='sum')(
              x * coord_mask, tx * coord_mask ) / 2.0
            loss_y = self.coord_scale * nn.MSELoss(reduction='sum')(
              y * coord_mask, ty * coord_mask ) / 2.0
            loss_w = self.coord_scale * nn.MSELoss(reduction='sum')(
              w * coord_mask, tw * coord_mask ) / 2.0
            loss_h = self.coord_scale * nn.MSELoss(reduction='sum')(
              h * coord_mask, th * coord_mask ) / 2.0
            loss_conf = nn.MSELoss(reduction='sum')(
              conf * conf_mask, tconf * conf_mask ) / 2.0
            loss_cls = self.class_scale * nn.CrossEntropyLoss(reduction='sum')(cls, tcls)
            loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls
            t4 = time.time()
            if False:
                print('-----------------------------------')
                print('        activation : %f' % (t1 - t0))
                print(' create pred_boxes : %f' % (t2 - t1))
                print('     build targets : %f' % (t3 - t2))
                print('       create loss : %f' % (t4 - t3))
                print('             total : %f' % (t4 - t0))
            print(
              '%d: nGT %d, recall %d, proposals %d, loss: x %f, y %f, w %f, h %f, conf %f, cls %f, total %f' % (
              self.seen, nGT, nCorrect, nProposals, loss_x.data[0], loss_y.data[0],
              loss_w.data[0], loss_h.data[0],loss_conf.data[0], loss_cls.data[0],
              loss.data[0])
            )
        return loss

    def build_targets(self, pred_boxes, target, nH, nW):
        anchors = self.anchors
        noobject_scale = self.noobject_scale
        object_scale = self.object_scale
        sil_thresh = self.thresh
        seen = self.seen

        nB = target.size(0)
        nA = self.n_anchors
        anchor_step = len(anchors) / n_anchors
        conf_mask = torch.ones(nB, nA, nH, nW) * noobject_scale
        coord_mask = torch.zeros(nB, nA, nH, nW)
        cls_mask = torch.zeros(nB, nA, nH, nW)
        tx = torch.zeros(nB, nA, nH, nW)
        ty = torch.zeros(nB, nA, nH, nW)
        tw = torch.zeros(nB, nA, nH, nW)
        th = torch.zeros(nB, nA, nH, nW)
        tconf = torch.zeros(nB, nA, nH, nW)
        tcls = torch.zeros(nB, nA, nH, nW)

        nAnchors = nA * nH * nW
        nPixels = nH * nW
        for b in range(nB):
            cur_pred_boxes = pred_boxes[b * nAnchors:(b + 1) * nAnchors].t()
            cur_ious = torch.zeros(nAnchors)
            for t in range(50):
                if target[b][t * 5 + 1] == 0:
                    break
                gx = target[b][t * 5 + 1] * nW
                gy = target[b][t * 5 + 2] * nH
                gw = target[b][t * 5 + 3] * nW
                gh = target[b][t * 5 + 4] * nH
                cur_gt_boxes = torch.FloatTensor([gx, gy, gw, gh]).repeat(nAnchors, 1).t()
                cur_ious = torch.max(cur_ious, bbox_ious(cur_pred_boxes, cur_gt_boxes, x1y1x2y2=False))
            conf_mask[b][cur_ious > sil_thresh] = 0
        if seen < 12800:
            if anchor_step == 4:
                tx = torch.FloatTensor(anchors).view(nA, anchor_step).index_select(1, torch.LongTensor([2])).view(1, nA, 1,
                                                                                                                  1).repeat(
                    nB, 1, nH, nW)
                ty = torch.FloatTensor(anchors).view(n_anchors, anchor_step).index_select(1, torch.LongTensor([2])).view(
                    1, nA, 1, 1).repeat(nB, 1, nH, nW)
            else:
                tx.fill_(0.5)
                ty.fill_(0.5)
            tw.zero_()
            th.zero_()
            coord_mask.fill_(1)

        nGT = 0
        nCorrect = 0
        for b in range(nB):
            for t in range(50):
                if target[b][t * 5 + 1] == 0:
                    break
                nGT = nGT + 1
                best_iou = 0.0
                best_n = -1
                min_dist = 10000
                gx = target[b][t * 5 + 1] * nW
                gy = target[b][t * 5 + 2] * nH
                gi = int(gx)
                gj = int(gy)
                gw = target[b][t * 5 + 3] * nW
                gh = target[b][t * 5 + 4] * nH
                gt_box = [0, 0, gw, gh]
                for n in range(nA):
                    aw = anchors[anchor_step * n]
                    ah = anchors[anchor_step * n + 1]
                    anchor_box = [0, 0, aw, ah]
                    iou = bbox_iou(anchor_box, gt_box, x1y1x2y2=False)
                    if anchor_step == 4:
                        ax = anchors[anchor_step * n + 2]
                        ay = anchors[anchor_step * n + 3]
                        dist = pow(((gi + ax) - gx), 2) + pow(((gj + ay) - gy), 2)
                    if iou > best_iou:
                        best_iou = iou
                        best_n = n
                    elif anchor_step == 4 and iou == best_iou and dist < min_dist:
                        best_iou = iou
                        best_n = n
                        min_dist = dist

                gt_box = [gx, gy, gw, gh]
                pred_box = pred_boxes[b * nAnchors + best_n * nPixels + gj * nW + gi]

                coord_mask[b][best_n][gj][gi] = 1
                cls_mask[b][best_n][gj][gi] = 1
                conf_mask[b][best_n][gj][gi] = object_scale
                tx[b][best_n][gj][gi] = target[b][t * 5 + 1] * nW - gi
                ty[b][best_n][gj][gi] = target[b][t * 5 + 2] * nH - gj
                tw[b][best_n][gj][gi] = math.log(gw / anchors[anchor_step * best_n])
                th[b][best_n][gj][gi] = math.log(gh / anchors[anchor_step * best_n + 1])
                iou = bbox_iou(gt_box, pred_box, x1y1x2y2=False)  # best_iou
                tconf[b][best_n][gj][gi] = iou
                tcls[b][best_n][gj][gi] = target[b][t * 5]
                if iou > 0.5:
                    nCorrect = nCorrect + 1

        return nGT, nCorrect, coord_mask, conf_mask, cls_mask, tx, ty, tw, th, tconf, tcls


class Artic_loss(nn.Module):
    def __init__( self, n_classes=2, n_anchors=3,
      device=None, batch=2, image_size=480 ):
        super(Artic_loss, self).__init__()
        self.device = device
        self.strides = [8, 16, 32]
        self.n_classes = n_classes
        self.n_anchors = n_anchors
        self.n_preds = 10 # 4 for regular yolo
        self.n_conf = self.n_preds + 1
        self.n_ch = (self.n_preds + 1 + self.n_classes)

        self.anchors = [[12, 16], [19, 36], [40, 28], [36, 75], [76, 55], [72, 146], [142, 110], [192, 243], [459, 401]]
        self.anch_masks = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        self.ignore_thre = 0.5

        self.masked_anchors, self.ref_anchors, self.grid_x, self.grid_y, self.anchor_w, self.anchor_h = [], [], [], [], [], []

        for i in range(3):
            all_anchors_grid = [(w / self.strides[i], h / self.strides[i]) for w, h in self.anchors]
            masked_anchors = np.array([all_anchors_grid[j] for j in self.anch_masks[i]], dtype=np.float32)
            ref_anchors = np.zeros((len(all_anchors_grid), 4), dtype=np.float32)
            ref_anchors[:, 2:] = np.array(all_anchors_grid, dtype=np.float32)
            ref_anchors = torch.from_numpy(ref_anchors)
            # calculate pred - xywh obj cls
            fsize = image_size // self.strides[i]
            grid_x = torch.arange(fsize, dtype=torch.float).repeat(
              batch, 3, nW, 1 ).to(device)
            grid_y = torch.arange(fsize, dtype=torch.float).repeat(
              batch, 3, nH, 1 ).permute(0, 1, 3, 2).to(device)
            anchor_w = torch.from_numpy(masked_anchors[:, 0]).repeat(
              batch, nH, nW, 1 ).permute(0, 3, 1, 2).to(device).unsqueeze(-1)
            anchor_h = torch.from_numpy(masked_anchors[:, 1]).repeat(
              batch, nH, nW, 1 ).permute(0, 3, 1, 2).to(device).unsqueeze(-1)

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
            fsize = output.shape[2]

            output = output.view(batchsize, self.n_anchors, self.n_ch, fsize, fsize)
            output = output.permute(0, 1, 3, 4, 2)  # .contiguous()

            """
            output shape = ( batchsize, anchors, fsize, fsize,
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
        truth_xs_all = labels[:, :, 0:-1:2] / self.strides[output_id]
        truth_ys_all = labels[:, :, 1:-1:2] / self.strides[output_id]
        truth_i_all = truth_x1_all.to(torch.int16).cpu().numpy()
        truth_j_all = truth_y1_all.to(torch.int16).cpu().numpy()

        for b in range(batchsize):
            n = int(nlabel[b])
            if n == 0:
                continue
            truth_box = torch.zeros(n, self.n_preds).to(self.device)
            import pdb; pdb.set_trace()
            truth_box[:n, 2:-1:2] = truth_xs_all[b, :n]
            truth_box[:n, 3:-1:2] = truth_ys_all[b, :n]
            truth_i = truth_i_all[b, :n]
            truth_j = truth_j_all[b, :n]

            # calculate iou between truth and reference anchors
            anchor_ious_all = bboxes_iou(truth_box.cpu(), self.ref_anchors[output_id], CIoU=True)

            # temp = bbox_iou(truth_box.cpu(), self.ref_anchors[output_id])

            best_n_all = anchor_ious_all.argmax(dim=1)
            best_n = best_n_all % 3
            best_n_mask = ( (best_n_all == self.anch_masks[output_id][0]) |
                            (best_n_all == self.anch_masks[output_id][1]) |
                            (best_n_all == self.anch_masks[output_id][2]) )

            if sum(best_n_mask) == 0:
                continue

            truth_box[:n, 0] = truth_x_all[b, :n]
            truth_box[:n, 1] = truth_y_all[b, :n]

            pred_ious = bboxes_iou(pred[b].view(-1, 4), truth_box, xyxy=False)
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
                    target[b, a, j, i, 0] = truth_x_all[b, ti] - truth_x_all[b, ti].to(torch.int16).to(torch.float)
                    target[b, a, j, i, 1] = truth_y_all[b, ti] - truth_y_all[b, ti].to(torch.int16).to(torch.float)
                    target[b, a, j, i, 2] = torch.log(
                        truth_w_all[b, ti] / torch.Tensor(self.masked_anchors[output_id])[best_n[ti], 0] + 1e-16)
                    target[b, a, j, i, 3] = torch.log(
                        truth_h_all[b, ti] / torch.Tensor(self.masked_anchors[output_id])[best_n[ti], 1] + 1e-16)
                    target[b, a, j, i, 4] = 1
                    target[b, a, j, i, 5 + labels[b, ti, 4].to(torch.int16).cpu().numpy()] = 1
                    tgt_scale[b, a, j, i, :] = torch.sqrt(2 - truth_w_all[b, ti] * truth_h_all[b, ti] / fsize / fsize)
        return obj_mask, tgt_mask, tgt_scale, target
