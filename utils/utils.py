from __future__ import division
import math
import time
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import random


def to_cpu(tensor):
    return tensor.detach().cpu()


def load_classes(path):
    """
    Loads class labels at 'path'
    """
    fp = open(path, "r")
    names = fp.read().split("\n")[:-1]
    return names
# def load_classes(path):
#     # Loads *.names file at 'path'
#     with open(path, 'r') as f:
#         names = f.read().split('\n')
#     return list(filter(None, names))  # filter removes empty strings (such as last line)


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def rescale_boxes(boxes, current_dim, original_shape):
    """ Rescales bounding boxes to the original shape """
    orig_h, orig_w = original_shape
    # The amount of padding that was added
    pad_x = max(orig_h - orig_w, 0) * (current_dim / max(original_shape))
    pad_y = max(orig_w - orig_h, 0) * (current_dim / max(original_shape))
    # Image height and width after padding is removed
    unpad_h = current_dim - pad_y
    unpad_w = current_dim - pad_x
    # Rescale bounding boxes to dimension of original image
    boxes[:, 0] = ((boxes[:, 0] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 1] = ((boxes[:, 1] - pad_y // 2) / unpad_h) * orig_h
    boxes[:, 2] = ((boxes[:, 2] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 3] = ((boxes[:, 3] - pad_y // 2) / unpad_h) * orig_h
    return boxes


# def xywh2xyxy(x):
#     y = x.new(x.shape)
#     y[..., 0] = x[..., 0] - x[..., 2] / 2
#     y[..., 1] = x[..., 1] - x[..., 3] / 2
#     y[..., 2] = x[..., 0] + x[..., 2] / 2
#     y[..., 3] = x[..., 1] + x[..., 3] / 2
#     return y

def xywh2xyxy(x):
    # Convert bounding box format from [x, y, w, h] to [x1, y1, x2, y2]
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def ap_per_class(tp, conf, pred_cls, target_cls):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (list).
        conf:  Objectness value from 0-1 (list).
        pred_cls: Predicted object classes (list).
        target_cls: True object classes (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)

    # Create Precision-Recall curve and compute AP for each class
    ap, p, r = [], [], []
    for c in tqdm.tqdm(unique_classes, desc="Computing AP"):
        i = pred_cls == c
        n_gt = (target_cls == c).sum()  # Number of ground truth objects
        n_p = i.sum()  # Number of predicted objects

        if n_p == 0 and n_gt == 0:
            continue
        elif n_p == 0 or n_gt == 0:
            ap.append(0)
            r.append(0)
            p.append(0)
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum()
            tpc = (tp[i]).cumsum()

            # Recall
            recall_curve = tpc / (n_gt + 1e-16)
            r.append(recall_curve[-1])

            # Precision
            precision_curve = tpc / (tpc + fpc)
            p.append(precision_curve[-1])

            # AP from recall-precision curve
            ap.append(compute_ap(recall_curve, precision_curve))

    # Compute F1 score (harmonic mean of precision and recall)
    p, r, ap = np.array(p), np.array(r), np.array(ap)
    f1 = 2 * p * r / (p + r + 1e-16)

    return p, r, ap, f1, unique_classes.astype("int32")


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def get_batch_statistics(outputs, targets, iou_threshold):
    """ Compute true positives, predicted scores and predicted labels per sample """
    batch_metrics = []
    for sample_i in range(len(outputs)):

        if outputs[sample_i] is None:
            continue

        output = outputs[sample_i]
        pred_boxes = output[:, :4]
        pred_scores = output[:, 4]
        pred_labels = output[:, -1]

        true_positives = np.zeros(pred_boxes.shape[0])

        annotations = targets[targets[:, 0] == sample_i][:, 1:]
        target_labels = annotations[:, 0] if len(annotations) else []
        if len(annotations):
            detected_boxes = []
            target_boxes = annotations[:, 1:]

            for pred_i, (pred_box, pred_label) in enumerate(zip(pred_boxes, pred_labels)):

                # If targets are found break
                if len(detected_boxes) == len(annotations):
                    break

                # Ignore if label is not one of the target labels
                if pred_label not in target_labels:
                    continue

                iou, box_index = bbox_iou(pred_box.unsqueeze(0), target_boxes).max(0)
                if iou >= iou_threshold and box_index not in detected_boxes:
                    true_positives[pred_i] = 1
                    detected_boxes += [box_index]
        batch_metrics.append([true_positives, pred_scores, pred_labels])
    return batch_metrics


def bbox_wh_iou(wh1, wh2):
    wh2 = wh2.t()
    w1, h1 = wh1[0], wh1[1]
    w2, h2 = wh2[0], wh2[1]
    inter_area = torch.min(w1, w2) * torch.min(h1, h2)
    union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area
    return inter_area / union_area


def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou


def non_max_suppression(prediction, conf_thres=0.5, nms_thres=0.4):
    """
    Removes detections with lower object confidence score than 'conf_thres' and performs
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """

    # From (center x, center y, width, height) to (x1, y1, x2, y2)
    prediction[..., :4] = xywh2xyxy(prediction[..., :4])
    output = [None for _ in range(len(prediction))]
    for image_i, image_pred in enumerate(prediction):
        # Filter out confidence scores below threshold
        image_pred = image_pred[image_pred[:, 4] >= conf_thres]
        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Object confidence times class confidence
        score = image_pred[:, 4] * image_pred[:, 5:].max(1)[0]
        # Sort by it
        image_pred = image_pred[(-score).argsort()]
        class_confs, class_preds = image_pred[:, 5:].max(1, keepdim=True)
        detections = torch.cat((image_pred[:, :5], class_confs.float(), class_preds.float()), 1)
        # Perform non-maximum suppression
        keep_boxes = []
        while detections.size(0):
            large_overlap = bbox_iou(detections[0, :4].unsqueeze(0), detections[:, :4]) > nms_thres
            label_match = detections[0, -1] == detections[:, -1]
            # Indices of boxes with lower confidence scores, large IOUs and matching labels
            invalid = large_overlap & label_match
            weights = detections[invalid, 4:5]
            # Merge overlapping bboxes by order of confidence
            detections[0, :4] = (weights * detections[invalid, :4]).sum(0) / weights.sum()
            keep_boxes += [detections[0]]
            detections = detections[~invalid]
        if keep_boxes:
            output[image_i] = torch.stack(keep_boxes)

    return output

# def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False):
#     # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
#     box2 = box2.t()
#
#     # Get the coordinates of bounding boxes
#     if x1y1x2y2:
#         # x1, y1, x2, y2 = box1
#         b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
#         b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
#     else:
#         # x, y, w, h = box1
#         b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
#         b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
#         b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
#         b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2
#
#     # Intersection area
#     inter_area = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
#                  (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)
#
#     # Union Area
#     union_area = ((b1_x2 - b1_x1) * (b1_y2 - b1_y1) + 1e-16) + \
#                  (b2_x2 - b2_x1) * (b2_y2 - b2_y1) - inter_area
#
#     iou = inter_area / union_area  # iou
#     if GIoU:  # Generalized IoU https://arxiv.org/pdf/1902.09630.pdf
#         c_x1, c_x2 = torch.min(b1_x1, b2_x1), torch.max(b1_x2, b2_x2)
#         c_y1, c_y2 = torch.min(b1_y1, b2_y1), torch.max(b1_y2, b2_y2)
#         c_area = (c_x2 - c_x1) * (c_y2 - c_y1)  # convex area
#         return iou - (c_area - union_area) / c_area  # GIoU
#
#     return iou

# def non_max_suppression2(prediction, conf_thres=0.5, nms_thres=0.5):
#     """
#     Removes detections with lower object confidence score than 'conf_thres'
#     Non-Maximum Suppression to further filter detections.
#     Returns detections with shape:
#         (x1, y1, x2, y2, object_conf, class_conf, class)
#     """
#
#     min_wh = 2  # (pixels) minimum box width and height
#
#     output = [None] * len(prediction)
#     for image_i, pred in enumerate(prediction):
#         # Experiment: Prior class size rejection
#         # x, y, w, h = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
#         # a = w * h  # area
#         # ar = w / (h + 1e-16)  # aspect ratio
#         # n = len(w)
#         # log_w, log_h, log_a, log_ar = torch.log(w), torch.log(h), torch.log(a), torch.log(ar)
#         # shape_likelihood = np.zeros((n, 60), dtype=np.float32)
#         # x = np.concatenate((log_w.reshape(-1, 1), log_h.reshape(-1, 1)), 1)
#         # from scipy.stats import multivariate_normal
#         # for c in range(60):
#         # shape_likelihood[:, c] =
#         #   multivariate_normal.pdf(x, mean=mat['class_mu'][c, :2], cov=mat['class_cov'][c, :2, :2])
#
#         # Multiply conf by class conf to get combined confidence
#         class_conf, class_pred = pred[:, 5:].max(1)
#         pred[:, 4] *= class_conf
#
#         # Select only suitable predictions
#         i = (pred[:, 4] > conf_thres) & (pred[:, 2:4] > min_wh).all(1) & torch.isfinite(pred).all(1)
#         pred = pred[i]
#
#         # If none are remaining => process next image
#         if len(pred) == 0:
#             continue
#
#         # Select predicted classes
#         class_conf = class_conf[i]
#         class_pred = class_pred[i].unsqueeze(1).float()
#
#         # Box (center x, center y, width, height) to (x1, y1, x2, y2)
#         pred[:, :4] = xywh2xyxy(pred[:, :4])
#         # pred[:, 4] *= class_conf  # improves mAP from 0.549 to 0.551
#
#         # Detections ordered as (x1y1x2y2, obj_conf, class_conf, class_pred)
#         pred = torch.cat((pred[:, :5], class_conf.unsqueeze(1), class_pred), 1)
#
#         # Get detections sorted by decreasing confidence scores
#         pred = pred[(-pred[:, 4]).argsort()]
#
#         det_max = []
#         nms_style = 'MERGE'  # 'OR' (default), 'AND', 'MERGE' (experimental)
#         for c in pred[:, -1].unique():
#             dc = pred[pred[:, -1] == c]  # select class c
#             n = len(dc)
#             if n == 1:
#                 det_max.append(dc)  # No NMS required if only 1 prediction
#                 continue
#             elif n > 100:
#                 dc = dc[:100]  # limit to first 100 boxes: https://github.com/ultralytics/yolov3/issues/117
#
#             # Non-maximum suppression
#             if nms_style == 'OR':  # default
#                 # METHOD1
#                 # ind = list(range(len(dc)))
#                 # while len(ind):
#                 # j = ind[0]
#                 # det_max.append(dc[j:j + 1])  # save highest conf detection
#                 # reject = (bbox_iou(dc[j], dc[ind]) > nms_thres).nonzero()
#                 # [ind.pop(i) for i in reversed(reject)]
#
#                 # METHOD2
#                 while dc.shape[0]:
#                     det_max.append(dc[:1])  # save highest conf detection
#                     if len(dc) == 1:  # Stop if we're at the last detection
#                         break
#                     iou = bbox_iou(dc[0], dc[1:])  # iou with other boxes
#                     dc = dc[1:][iou < nms_thres]  # remove ious > threshold
#
#             elif nms_style == 'AND':  # requires overlap, single boxes erased
#                 while len(dc) > 1:
#                     iou = bbox_iou(dc[0], dc[1:])  # iou with other boxes
#                     if iou.max() > 0.5:
#                         det_max.append(dc[:1])
#                     dc = dc[1:][iou < nms_thres]  # remove ious > threshold
#
#             elif nms_style == 'MERGE':  # weighted mixture box
#                 while len(dc):
#                     if len(dc) == 1:
#                         det_max.append(dc)
#                         break
#                     i = bbox_iou(dc[0], dc) > nms_thres  # iou with other boxes
#                     weights = dc[i, 4:5]
#                     dc[0, :4] = (weights * dc[i, :4]).sum(0) / weights.sum()
#                     det_max.append(dc[:1])
#                     dc = dc[i == 0]
#
#             elif nms_style == 'SOFT':  # soft-NMS https://arxiv.org/abs/1704.04503
#                 sigma = 0.5  # soft-nms sigma parameter
#                 while len(dc):
#                     if len(dc) == 1:
#                         det_max.append(dc)
#                         break
#                     det_max.append(dc[:1])
#                     iou = bbox_iou(dc[0], dc[1:])  # iou with other boxes
#                     dc = dc[1:]
#                     dc[:, 4] *= torch.exp(-iou ** 2 / sigma)  # decay confidences
#
#         if len(det_max):
#             det_max = torch.cat(det_max)  # concatenate
#             output[image_i] = det_max[(-det_max[:, 4]).argsort()]  # sort
#
#     return output

def common_entries(*dcts):
    for i in set(dcts[0]).intersection(*dcts[1:]):
        yield (i,) + tuple(d[i] for d in dcts)

def plot_images(imgs, targets, fname='images.jpg'):
    # Plots training images overlaid with targets
    imgs = imgs.cpu().numpy()
    targets = targets.cpu().numpy()

    fig = plt.figure(figsize=(10, 10))
    bs, _, h, w = imgs.shape  # batch size, _, height, width
    ns = np.ceil(bs ** 0.5)  # number of subplots

    for i in range(bs):
        boxes = xywh2xyxy(targets[targets[:, 0] == i, 2:6]).T
        # boxes = targets[targets[:, 0] == i, 2:6]
        boxes[[0, 2]] *= w
        boxes[[1, 3]] *= h
        plt.subplot(ns, ns, i + 1).imshow(imgs[i].transpose(1, 2, 0))
        plt.plot(boxes[[0, 2, 2, 0, 0]], boxes[[1, 1, 3, 3, 1]], '.-')
        plt.axis('off')
    fig.tight_layout()
    fig.savefig(fname, dpi=300)
    plt.close()


def build_targets(pred_boxes, pred_cls, target, anchors, ignore_thres):

    ByteTensor = torch.cuda.ByteTensor if pred_boxes.is_cuda else torch.ByteTensor
    FloatTensor = torch.cuda.FloatTensor if pred_boxes.is_cuda else torch.FloatTensor

    nB = pred_boxes.size(0)
    nA = pred_boxes.size(1)
    nC = pred_cls.size(-1)
    nG = pred_boxes.size(2)

    # Output tensors
    obj_mask = ByteTensor(nB, nA, nG, nG).fill_(0)
    noobj_mask = ByteTensor(nB, nA, nG, nG).fill_(1)
    class_mask = FloatTensor(nB, nA, nG, nG).fill_(0)
    iou_scores = FloatTensor(nB, nA, nG, nG).fill_(0)
    tx = FloatTensor(nB, nA, nG, nG).fill_(0)
    ty = FloatTensor(nB, nA, nG, nG).fill_(0)
    tw = FloatTensor(nB, nA, nG, nG).fill_(0)
    th = FloatTensor(nB, nA, nG, nG).fill_(0)
    tcls = FloatTensor(nB, nA, nG, nG, nC).fill_(0)

    # Convert to position relative to box
    target_boxes = target[:, 2:6] * nG
    gxy = target_boxes[:, :2]
    gwh = target_boxes[:, 2:]
    # Get anchors with best iou
    ious = torch.stack([bbox_wh_iou(anchor, gwh) for anchor in anchors])
    best_ious, best_n = ious.max(0)
    # Separate target values
    b, target_labels = target[:, :2].long().t()
    gx, gy = gxy.t()
    gw, gh = gwh.t()
    gi, gj = gxy.long().t()
    # Set masks
    obj_mask[b, best_n, gj, gi] = 1
    noobj_mask[b, best_n, gj, gi] = 0

    # Set noobj mask to zero where iou exceeds ignore threshold
    for i, anchor_ious in enumerate(ious.t()):
        noobj_mask[b[i], anchor_ious > ignore_thres, gj[i], gi[i]] = 0

    # Coordinates
    tx[b, best_n, gj, gi] = gx - gx.floor()
    ty[b, best_n, gj, gi] = gy - gy.floor()
    # Width and height
    tw[b, best_n, gj, gi] = torch.log(gw / anchors[best_n][:, 0] + 1e-16)
    th[b, best_n, gj, gi] = torch.log(gh / anchors[best_n][:, 1] + 1e-16)
    # One-hot encoding of label
    tcls[b, best_n, gj, gi, target_labels] = 1
    # Compute label correctness and iou at best anchor
    class_mask[b, best_n, gj, gi] = (pred_cls[b, best_n, gj, gi].argmax(-1) == target_labels).float()
    iou_scores[b, best_n, gj, gi] = bbox_iou(pred_boxes[b, best_n, gj, gi], target_boxes, x1y1x2y2=False)

    tconf = obj_mask.float()
    return iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf


def plot_one_box(x1, y1, x2, y2, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * max(img.shape[0:2])) + 1  # line thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x1), int(y1)), (int(x2), int(y2))
    cv2.rectangle(img, c1, c2, color, thickness=tl)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)