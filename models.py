from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from collections import OrderedDict
import argparse

from utils.parse_config import *
from utils.utils import build_targets, to_cpu, common_items, non_max_suppression, rng2inds
import os

import matplotlib.pyplot as plt
import matplotlib.patches as patches


# def create_modules(module_defs):
#     """
#     Constructs module list of layer blocks from module configuration in module_defs
#     """
#     hyperparams = module_defs.pop(0)
#     output_filters = [int(hyperparams["channels"])]
#     module_list = nn.ModuleList()
#     for module_i, module_def in enumerate(module_defs):
#         modules = nn.Sequential()
#
#         if module_def["type"] == "convolutional":
#             bn = int(module_def["batch_normalize"])
#             filters = int(module_def["filters"])
#             kernel_size = int(module_def["size"])
#             pad = (kernel_size - 1) // 2
#             modules.add_module(
#                 f"conv_{module_i}",
#                 nn.Conv2d(
#                     in_channels=output_filters[-1],
#                     out_channels=filters,
#                     kernel_size=kernel_size,
#                     stride=int(module_def["stride"]),
#                     padding=pad,
#                     bias=not bn,
#                 ),
#             )
#             if bn:
#                 modules.add_module(f"batch_norm_{module_i}", nn.BatchNorm2d(filters, momentum=0.9, eps=1e-5))
#             if module_def["activation"] == "leaky":
#                 modules.add_module(f"leaky_{module_i}", nn.LeakyReLU(0.1))
#         elif module_def["type"] == "dropout":
#             modules.add_module(f"dropout_{module_i}", nn.Dropout(float(module_def["prob"])))
#         elif module_def["type"] == "maxpool":
#             kernel_size = int(module_def["size"])
#             stride = int(module_def["stride"])
#             if kernel_size == 2 and stride == 1:
#                 modules.add_module(f"_debug_padding_{module_i}", nn.ZeroPad2d((0, 1, 0, 1)))
#             maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=int((kernel_size - 1) // 2))
#             modules.add_module(f"maxpool_{module_i}", maxpool)
#
#         elif module_def["type"] == "upsample":
#             upsample = Upsample(scale_factor=int(module_def["stride"]), mode="nearest")
#             modules.add_module(f"upsample_{module_i}", upsample)
#
#         elif module_def["type"] == "route":
#             layers = [int(x) for x in module_def["layers"].split(",")]
#             filters = sum([output_filters[1:][i] for i in layers])
#             modules.add_module(f"route_{module_i}", EmptyLayer())
#
#         elif module_def["type"] == "shortcut":
#             filters = output_filters[1:][int(module_def["from"])]
#             modules.add_module(f"shortcut_{module_i}", EmptyLayer())
#
#         elif module_def["type"] == "yolo":
#             anchor_idxs = [int(x) for x in module_def["mask"].split(",")]
#             # Extract anchors
#             anchors = [int(x) for x in module_def["anchors"].split(",")]
#             anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
#             anchors = [anchors[i] for i in anchor_idxs]
#             num_classes = int(module_def["classes"])
#             img_size = int(hyperparams["height"])
#             # Define detection layer
#             yolo_layer = YOLOLayer(anchors, num_classes, img_size)
#             modules.add_module(f"yolo_{module_i}", yolo_layer)
#         # Register module list and number of output filters
#         module_list.append(modules)
#         output_filters.append(filters)
#
#     return hyperparams, module_list
def create_modules(hyperparams, sx_module_defs):
    """
    Constructs module list of layer blocks from module configuration in module_defs
    """
    # hyperparams = module_defs.pop(0)

    if not 'streams' in hyperparams:
        hyperparams['streams'] = '0'

    sx_output_filters = OrderedDict()
    sx_module_lists = OrderedDict()
    for sx, _ in sx_module_defs.items():
        sx_output_filters[sx] = [int(hyperparams['channels'])] if sx in hyperparams['streams'] else []  # TODO: make variable nb of channels in each stream
        sx_module_lists[sx] = []
    yolo_index = -1

    for sx, module_defs in sx_module_defs.items():
        filters = None

        for i, module_def in enumerate(module_defs):
            modules = nn.Sequential()

            if module_def['type'] == 'convolutional':
                bn = int(module_def['batch_normalize'])
                filters = int(module_def['filters'])
                kernel_size = int(module_def['size'])
                pad = (kernel_size - 1) // 2 if int(module_def['pad']) else 0
                modules.add_module('conv@%s_%d' % (sx, i), nn.Conv2d(in_channels=sx_output_filters[sx][-1],
                                                            out_channels=filters,
                                                            kernel_size=kernel_size,
                                                            stride=int(module_def['stride']),
                                                            padding=pad,
                                                            bias=not bn))
                if bn:
                    modules.add_module('batch_norm@%s_%d' % (sx, i), nn.BatchNorm2d(filters))
                if module_def['activation'] == 'leaky':
                    modules.add_module('leaky@%s_%d' % (sx, i), nn.LeakyReLU(0.1, inplace=True))

            elif module_def['type'] == 'rconvolutional':
                bn = int(module_def['batch_normalize'])
                if bn:
                    modules.add_module('batch_norm@%s_%d' % (sx, i), nn.BatchNorm2d(sx_output_filters[sx][-1]))
                if module_def['activation'] == 'leaky':
                    modules.add_module('leaky@%s_%d' % (sx, i), nn.LeakyReLU(0.1, inplace=True))

                kernel_size = int(module_def['size'])
                pad = (kernel_size - 1) // 2 if int(module_def['pad']) else 0
                filters = int(module_def['filters'])
                modules.add_module('conv@%s_%d' % (sx, i), nn.Conv2d(in_channels=sx_output_filters[sx][-1],
                                                                     out_channels=filters,
                                                                     kernel_size=kernel_size,
                                                                     stride=int(module_def['stride']),
                                                                     padding=pad,
                                                                     bias=not bn))
            elif module_def['type'] == 'maxpool':
                kernel_size = int(module_def['size'])
                stride = int(module_def['stride'])
                if kernel_size == 2 and stride == 1:
                    modules.add_module('_debug_padding@%s_%d' % (sx, i), nn.ZeroPad2d((0, 1, 0, 1)))
                maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=int((kernel_size - 1) // 2))
                modules.add_module('maxpool@%s_%d' % (sx, i), maxpool)

            elif module_def['type'] == 'upsample':
                upsample = nn.Upsample(scale_factor=int(module_def['stride']), mode='nearest')
                modules.add_module('upsample@%s_%d' % (sx, i), upsample)

            elif module_def['type'] == 'route':
                layers = []
                for x in module_def['layers'].strip().split(','):
                    x_spl = x.strip().split('@')
                    layers.append((sx, int(x_spl[0])) if len(x_spl) == 1 else (x_spl[1], int(x_spl[0])))

                filters = sum([sx_output_filters[sx_l][nl + 1 if nl > 0 else nl] for sx_l,nl in layers])
                modules.add_module('route@%s_%d' % (sx, i), EmptyLayer())

            elif module_def['type'] == 'dropout':
                modules.add_module('dropout@%s_%d' % (sx, i), nn.Dropout(float(module_def['prob'])))

            elif module_def['type'] == 'shortcut':
                filters = sx_output_filters[sx][int(module_def['from'])]
                modules.add_module('shortcut@%s_%d' % (sx, i), EmptyLayer())

            elif module_def['type'] == 'activation':
                if module_def['function'] == 'leaky':
                    modules.add_module('leaky@%s_%d' % (sx, i), nn.LeakyReLU(0.1, inplace=True))

            elif module_def['type'] == 'bn':
                modules.add_module('batch_norm@%s_%d' % (sx, i), nn.BatchNorm2d(sx_output_filters[sx][-1]))

            elif module_def['type'] == 'yolo':
                yolo_index += 1
                anchor_idxs = [int(x) for x in module_def['mask'].split(',')]
                # Extract anchors
                anchors = [float(x) for x in module_def['anchors'].split(',')]
                anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
                anchors = [anchors[i] for i in anchor_idxs]
                nc = int(module_def['classes'])  # number of classes
                img_size = hyperparams['height']
                # Define detection layer
                modules.add_module('yolo@%s_%d' % (sx, i), YOLOLayer(anchors, nc, img_size))

            # Register module list and number of output filters
            sx_module_lists[sx].append(modules)
            if filters:
                sx_output_filters[sx].append(filters)

    return sx_module_lists


class Upsample(nn.Module):
    """ nn.Upsample is deprecated """

    def __init__(self, scale_factor, mode="nearest"):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        return x


class EmptyLayer(nn.Module):
    """Placeholder for 'route' and 'shortcut' layers"""

    def __init__(self):
        super(EmptyLayer, self).__init__()


class YOLOLayer(nn.Module):
    """Detection layer"""

    def __init__(self, anchors, num_classes, img_dim=416):
        super(YOLOLayer, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.ignore_thres = 0.5
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.obj_scale = 1
        self.noobj_scale = 100
        self.metrics = {}
        self.img_dim = img_dim
        self.grid_size = 0  # grid size

    def compute_grid_offsets(self, grid_size, cuda=True):
        self.grid_size = grid_size
        g = self.grid_size
        FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        self.stride = self.img_dim / self.grid_size
        # Calculate offsets for each grid
        self.grid_x = torch.arange(g).repeat(g, 1).view([1, 1, g, g]).type(FloatTensor)
        self.grid_y = torch.arange(g).repeat(g, 1).t().view([1, 1, g, g]).type(FloatTensor)
        self.scaled_anchors = FloatTensor([(a_w / self.stride, a_h / self.stride) for a_w, a_h in self.anchors])
        self.anchor_w = self.scaled_anchors[:, 0:1].view((1, self.num_anchors, 1, 1))
        self.anchor_h = self.scaled_anchors[:, 1:2].view((1, self.num_anchors, 1, 1))

    def forward(self, x, targets=None, img_dim=None):

        # Tensors for cuda support
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
        ByteTensor = torch.cuda.ByteTensor if x.is_cuda else torch.ByteTensor

        self.img_dim = img_dim
        num_samples = x.size(0)
        grid_size = x.size(2)

        prediction = (
            x.view(num_samples, self.num_anchors, self.num_classes + 5, grid_size, grid_size)
            .permute(0, 1, 3, 4, 2)
            .contiguous()
        )

        # Get outputs
        x = torch.sigmoid(prediction[..., 0])  # Center x
        y = torch.sigmoid(prediction[..., 1])  # Center y
        w = prediction[..., 2]  # Width
        h = prediction[..., 3]  # Height
        pred_conf = torch.sigmoid(prediction[..., 4])  # Conf
        pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls pred.

        # If grid size does not match current we compute new offsets
        if grid_size != self.grid_size:
            self.compute_grid_offsets(grid_size, cuda=x.is_cuda)

        # Add offset and scale with anchors
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = x.data + self.grid_x
        pred_boxes[..., 1] = y.data + self.grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * self.anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * self.anchor_h

        output = torch.cat(
            (
                pred_boxes.view(num_samples, -1, 4) * self.stride,
                pred_conf.view(num_samples, -1, 1),
                pred_cls.view(num_samples, -1, self.num_classes),
            ),
            -1,
        )

        if targets is None:
            return output, 0, 0
        else:
            iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf = build_targets(
                pred_boxes=pred_boxes,
                pred_cls=pred_cls,
                target=targets,
                anchors=self.scaled_anchors,
                ignore_thres=self.ignore_thres,
            )

            # Loss : Mask outputs to ignore non-existing objects (except with conf. loss)
            loss_x = self.mse_loss(x[obj_mask], tx[obj_mask])
            loss_y = self.mse_loss(y[obj_mask], ty[obj_mask])
            loss_w = self.mse_loss(w[obj_mask], tw[obj_mask])
            loss_h = self.mse_loss(h[obj_mask], th[obj_mask])
            loss_conf_obj = self.bce_loss(pred_conf[obj_mask], tconf[obj_mask])
            loss_conf_noobj = self.bce_loss(pred_conf[noobj_mask], tconf[noobj_mask])
            loss_conf = self.obj_scale * loss_conf_obj + self.noobj_scale * loss_conf_noobj
            loss_cls = self.bce_loss(pred_cls[obj_mask], tcls[obj_mask])
            total_loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls

            # Metrics
            cls_acc = 100 * class_mask[obj_mask].mean()
            conf_obj = pred_conf[obj_mask].mean()
            conf_noobj = pred_conf[noobj_mask].mean()
            conf50 = (pred_conf > 0.5).float()
            iou50 = (iou_scores > 0.5).float()
            iou75 = (iou_scores > 0.75).float()
            detected_mask = conf50 * class_mask * tconf
            precision = torch.sum(iou50 * detected_mask) / (conf50.sum() + 1e-16)
            recall50 = torch.sum(iou50 * detected_mask) / (obj_mask.sum() + 1e-16)
            recall75 = torch.sum(iou75 * detected_mask) / (obj_mask.sum() + 1e-16)

            self.metrics = {
                "loss": to_cpu(total_loss).item(),
                "x": to_cpu(loss_x).item(),
                "y": to_cpu(loss_y).item(),
                "w": to_cpu(loss_w).item(),
                "h": to_cpu(loss_h).item(),
                "conf": to_cpu(loss_conf).item(),
                "cls": to_cpu(loss_cls).item(),
                "cls_acc": to_cpu(cls_acc).item(),
                "recall50": to_cpu(recall50).item(),
                "recall75": to_cpu(recall75).item(),
                "precision": to_cpu(precision).item(),
                "conf_obj": to_cpu(conf_obj).item(),
                "conf_noobj": to_cpu(conf_noobj).item(),
                "grid_size": grid_size,
            }

        loss_xy = loss_x + loss_y
        loss_wh = loss_w + loss_h
        return output, total_loss, torch.stack((loss_xy, loss_wh, loss_conf, loss_cls, total_loss)).detach()

# class Darknet(nn.Module):
#     """YOLOv3 object detection model"""
#
#     def __init__(self, config_path, img_size=416):
#         super(Darknet, self).__init__()
#         self.module_defs = parse_model_config(config_path)
#         self.hyperparams, self.module_list = create_modules(self.module_defs)
#         self.yolo_layers = [layer[0] for layer in self.module_list if hasattr(layer[0], "metrics")]
#         self.img_size = img_size
#         self.seen = 0
#         self.header_info = np.array([0, 0, 0, self.seen, 0], dtype=np.int32)
#
#     def forward(self, x, targets=None):
#         img_dim = x.shape[2]
#         loss = 0
#         layer_outputs, yolo_outputs = [], []
#         for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
#             if module_def["type"] in ["convolutional", "upsample", "maxpool", "dropout"]:
#                 x = module(x)
#             elif module_def["type"] == "route":
#                 x = torch.cat([layer_outputs[int(layer_i)] for layer_i in module_def["layers"].split(",")], 1)
#             elif module_def["type"] == "shortcut":
#                 layer_i = int(module_def["from"])
#                 x = layer_outputs[-1] + layer_outputs[layer_i]
#             elif module_def["type"] == "yolo":
#                 x, layer_loss = module[0](x, targets, img_dim)
#                 loss += layer_loss
#                 yolo_outputs.append(x)
#             layer_outputs.append(x)
#         yolo_outputs = to_cpu(torch.cat(yolo_outputs, 1))
#
#         return yolo_outputs if targets is None else (loss, yolo_outputs)
#
#     def load_darknet_weights(self, weights_path):
#         """Parses and loads the weights stored in 'weights_path'"""
#
#         # Open the weights file
#         with open(weights_path, "rb") as f:
#             header = np.fromfile(f, dtype=np.int32, count=5)  # First five are header values
#             self.header_info = header  # Needed to write header when saving weights
#             self.seen = header[3]  # number of images seen during training
#             weights = np.fromfile(f, dtype=np.float32)  # The rest are weights
#
#         # Establish cutoff for loading backbone weights
#         cutoff = None
#         if "darknet53.conv.74" in weights_path:
#             cutoff = 75  # TODO: adjust number to layers that need initialization, not all of them
#         elif 'yolov3-tiny.conv.15' in weights_path:
#             cutoff = 9
#
#         c = 0
#         ptr = 0
#         for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
#             if c == cutoff:
#                 break
#             if module_def["type"] == "convolutional":
#                 conv_layer = module[0]
#                 c += 1
#                 if module_def["batch_normalize"]:
#                     # Load BN bias, weights, running mean and running variance
#                     bn_layer = module[1]
#                     num_b = bn_layer.bias.numel()  # Number of biases
#                     # Bias
#                     bn_b = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.bias)
#                     bn_layer.bias.data.copy_(bn_b)
#                     ptr += num_b
#                     # Weight
#                     bn_w = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.weight)
#                     bn_layer.weight.data.copy_(bn_w)
#                     ptr += num_b
#                     # Running Mean
#                     bn_rm = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.running_mean)
#                     bn_layer.running_mean.data.copy_(bn_rm)
#                     ptr += num_b
#                     # Running Var
#                     bn_rv = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.running_var)
#                     bn_layer.running_var.data.copy_(bn_rv)
#                     ptr += num_b
#                 else:
#                     # Load conv. bias
#                     num_b = conv_layer.bias.numel()
#                     conv_b = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(conv_layer.bias)
#                     conv_layer.bias.data.copy_(conv_b)
#                     ptr += num_b
#                 # Load conv. weights
#                 num_w = conv_layer.weight.numel()
#                 conv_w = torch.from_numpy(weights[ptr : ptr + num_w]).view_as(conv_layer.weight)
#                 conv_layer.weight.data.copy_(conv_w)
#                 ptr += num_w
#
#     def save_darknet_weights(self, path, cutoff=-1):
#         """
#             @:param path    - path of the new weights file
#             @:param cutoff  - save layers between 0 and cutoff (cutoff = -1 -> all are saved)
#         """
#         fp = open(path, "wb")
#         self.header_info[3] = self.seen
#         self.header_info.tofile(fp)
#
#         # Iterate through layers
#         for i, (module_def, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
#             if module_def["type"] == "convolutional":
#                 conv_layer = module[0]
#                 # If batch norm, load bn first
#                 if module_def["batch_normalize"]:
#                     bn_layer = module[1]
#                     bn_layer.bias.data.cpu().numpy().tofile(fp)
#                     bn_layer.weight.data.cpu().numpy().tofile(fp)
#                     bn_layer.running_mean.data.cpu().numpy().tofile(fp)
#                     bn_layer.running_var.data.cpu().numpy().tofile(fp)
#                 # Load conv bias
#                 else:
#                     conv_layer.bias.data.cpu().numpy().tofile(fp)
#                 # Load conv weights
#                 conv_layer.weight.data.cpu().numpy().tofile(fp)
#
#         fp.close()

# def get_yolo_layers(model):
#     '''
#     List YOLO layers in stream 0.
#     Question: can I have yolo layers in different streams. if so, make changes here (TODO).
#     :param model:
#     :return:
#     '''
#     a = [module_def['type'] == 'yolo' for module_def in model.sx_module_defs['0']]
#     return [i for i, x in enumerate(a) if x]  # [82, 94, 106] for yolov3


def get_yolo_layers(model):
    '''
    List YOLO layers in stream 0.
    Question: can I have yolo layers in different streams. if so, make changes here (TODO).
    :param model:
    :return:
    '''

    yolo_indices = dict()
    for sx, defs_sx in model.sx_module_defs.items():
        yolo_layers = [module_def['type'] == 'yolo' for module_def in defs_sx]
        yolo_indices[sx] = [i for i, x in enumerate(yolo_layers) if x]  # [82, 94, 106] for yolov3

    return yolo_indices

class Darknet(nn.Module):
    """YOLOv3 object detection model"""

    def __init__(self, config_path, img_size=416):
        super(Darknet, self).__init__()
        hyp, self.sx_module_defs = parse_model_cfg(config_path)
        self.sx_module_lists = create_modules(hyp, self.sx_module_defs) # pylist of pylists (not nn.ModuleList)
        self.yolo_layers = get_yolo_layers(self)
        self.img_size = img_size  # not used?
        self.seen = 0
        self.header_info = np.array([0, 0, 0, self.seen, 0], dtype=np.int32)
        # create an actual nn.ModuleList (so it registers model parameters, so optimizer "sees" them)
        self.module_list = nn.ModuleList([m for _, sublist in self.sx_module_lists.items() for m in sublist])

        hyp['streams'] = hyp['streams'].split(',') if 'streams' in hyp else ['0']
        self.hyperparams = hyp


    def forward(self, x0, targets=None):
        img_dim = max(x0[0].shape[-2:])
        loss = 0
        loss_items = None
        yolos = dict()
        layer_outputs = {sx: [] for sx in self.sx_module_lists.keys()}

        x = {sx: (x0[i] if sx in self.hyperparams['streams'] else None)
             for i, sx in enumerate(self.sx_module_defs.keys())}

        for sx, defs_sx in self.sx_module_defs.items():
            modules_sx = self.sx_module_lists[sx]
            yolos[sx] = []
            for i, (module_def, module) in enumerate(zip(defs_sx, modules_sx)):
                if module_def["type"] in ["convolutional", "upsample", "maxpool", "dropout", "bn", "activation", "rconvolutional"]:
                    x[sx] = module(x[sx])
                elif module_def["type"] == "route":
                    routes = [x for x in module_def['layers'].split(',')]
                    if len(routes) == 1:
                        r_spl = routes[0].split('@')
                        sx_i, lx_i = (sx, int(r_spl[0])) if len(r_spl) == 1 else (r_spl[1], int(r_spl[0]))
                        x[sx] = layer_outputs[sx_i][lx_i]
                    else:
                        layer_cat = []
                        for r in routes:
                            r_spl = r.split('@')
                            sx_i, lx_i = ((sx, int(r_spl[0])) if len(r_spl) == 1 else (r_spl[1], int(r_spl[0])))
                            layer_cat.append(layer_outputs[sx_i][lx_i])
                        x[sx] = torch.cat(layer_cat, 1)
                elif module_def["type"] == "shortcut":
                    layer_i = int(module_def["from"])
                    x[sx] = layer_outputs[sx][-1] + layer_outputs[sx][layer_i]
                elif module_def["type"] == "yolo":
                    x[sx], layer_loss, layer_loss_items = module[0](x[sx], targets, img_dim)
                    loss += layer_loss
                    loss_items = (layer_loss_items if loss_items is None else (loss_items + layer_loss_items))
                    yolos[sx].append(x[sx])
                layer_outputs[sx].append(x[sx])

        yolo_outputs = []
        for _, yolos_sx in yolos.items():
            # yolo_outputs += [torch.stack(yolos_sx).sum(dim=0) / len(yolos_sx)]
            yolo_outputs += yolos_sx
        yolo_outputs = to_cpu(torch.cat(yolo_outputs, 1))

        # yolo_outputs = []
        # for sx, yolos_sx in yolos.items():
        #     yolo_outputs += [to_cpu(torch.cat(yolos_sx, 1))]

        return yolo_outputs if targets is None else (loss, loss_items, yolo_outputs)

    def load_darknet_weights(self, weights_path, k=0, cutoff=-1, initial_freeze=True):
        """Parses and loads the weights stored in 'weights_path'"""

        if 'yolov3-tiny.conv.15' in weights_path:
            cutoff = 9
            # Open the weiheader[-1]coghts file
            with open(weights_path, "rb") as f:
                self.header_info = np.fromfile(f, dtype=np.int32, count=5)  # First five are header values
                weights = np.fromfile(f, dtype=np.float32)  # The rest are weights
        else:
            # This code's weight file structures
            with open(weights_path, "rb") as f:
                header = np.fromfile(f, dtype=np.int32, count=(5+1))
                self.header_info = header[:-1]  # Needed to write header when saving weights
                self.seen = header[3]  # number of images seen during training
                self.streams_len = np.fromfile(f, dtype=np.int32, count=header[-1])
                weights = np.fromfile(f, dtype=np.float32)

        # Establish cutoff for loading backbone weights
        # cutoff = None
        # if "darknet53.conv.74" in weights_path:
        #     cutoff = 75  # TODO: adjust number to layers that need initialization, not all of them

        c = 0
        ptr = 0

        sx = self.hyperparams['streams'][k]
        defs_sx = self.sx_module_defs[sx]
        modules_sx = self.sx_module_lists[sx]

        for i, (module_def, module) in enumerate(zip(defs_sx, modules_sx)):
            if c == cutoff:
                break
            if module_def["type"] == "convolutional":
                conv_layer = module[0]
                c += 1
                if module_def["batch_normalize"]:
                    # Load BN bias, weights, running mean and running variance
                    bn_layer = module[1]
                    num_b = bn_layer.bias.numel()  # Number of biases
                    # Bias
                    bn_b = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.bias)
                    bn_layer.bias.data.copy_(bn_b)
                    ptr += num_b
                    # Weight
                    bn_w = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.weight)
                    bn_layer.weight.data.copy_(bn_w)
                    ptr += num_b
                    # Running Mean
                    bn_rm = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.running_mean)
                    bn_layer.running_mean.data.copy_(bn_rm)
                    ptr += num_b
                    # Running Var
                    bn_rv = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.running_var)
                    bn_layer.running_var.data.copy_(bn_rv)
                    ptr += num_b
                else:
                    # Load conv. bias
                    num_b = conv_layer.bias.numel()
                    conv_b = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(conv_layer.bias)
                    conv_layer.bias.data.copy_(conv_b)
                    ptr += num_b
                # Load conv. weights
                num_w = conv_layer.weight.numel()
                conv_w = torch.from_numpy(weights[ptr : ptr + num_w]).view_as(conv_layer.weight)
                conv_layer.weight.data.copy_(conv_w)
                ptr += num_w

                if initial_freeze:
                    for p in module.parameters():
                        p.requires_grad = False


    # def save_darknet_weights(self, path, cutoff=None, ff=False): #, discard_root=False):
    #     """
    #         @:param path    - path of the new weights file
    #         @:param cutoff  - save layers between 0 and cutoff (cutoff = -1 -> all are saved)
    #     """
    #
    #     # cutoff needs to be: (a) a scalar, (b) a dict, or (c) None.
    #     if cutoff is not None and not hasattr(cutoff, '__len__'):  # if a scalar
    #         cutoff = {'0': cutoff}
    #     elif isinstance(cutoff, dict):
    #         assert len(cutoff) == len(self.sx_module_defs)  # error check
    #     elif cutoff is None:
    #         cutoff = {sx: -1 for sx in self.sx_module_defs.keys()}
    #     else:
    #         raise AttributeError
    #
    #     conv_modules = []
    #     for sx, modules_sx in self.sx_module_defs.items():
    #         cutoff_list = [module_def['type'] == 'convolutional' for i, module_def in enumerate(modules_sx)
    #                        if cutoff[sx] == -1 or i < cutoff[sx]]
    #         if sx != '0':
    #             conv_modules += [np.sum(cutoff_list)]
    #         elif not ff:
    #             conv_modules += [np.sum(cutoff_list)]
    #         else: # ff
    #             conv_modules[-1] += np.sum(cutoff_list)
    #
    #     self.lengths_sx = np.array([len(conv_modules)] + conv_modules, dtype=np.int32)
    #     self.header_info[3] = self.seen
    #
    #     fp = open(path, "wb")
    #     np.concatenate([self.header_info, self.lengths_sx]).tofile(fp)
    #
    #     # Iterate through layers
    #     root_defs, root_module_list = [], []
    #     if ff and '0' in self.sx_module_lists:
    #         root_defs, root_module_list = self.sx_module_defs['0'], self.sx_module_lists['0']
    #         del self.sx_module_defs['0']
    #         del self.sx_module_lists['0']
    #
    #     for sx, defs_sx in self.sx_module_defs.items():
    #         defs_sx = self.sx_module_defs[sx] + root_defs if sx != '0' else self.sx_module_defs[sx]
    #         modules_sx = self.sx_module_lists[sx] + root_module_list if sx != '0' else self.sx_module_lists[sx]
    #         for i, (module_def, module) in enumerate(zip(defs_sx[:cutoff[sx]], modules_sx[:cutoff[sx]])):
    #             if module_def["type"] == "convolutional":
    #                 conv_layer = module[0]
    #                 # If batch norm, load bn first
    #                 if module_def["batch_normalize"]:
    #                     bn_layer = module[1]
    #                     bn_layer.bias.data.cpu().numpy().tofile(fp)
    #                     bn_layer.weight.data.cpu().numpy().tofile(fp)
    #                     bn_layer.running_mean.data.cpu().numpy().tofile(fp)
    #                     bn_layer.running_var.data.cpu().numpy().tofile(fp)
    #                 # Load conv bias
    #                 else:
    #                     conv_layer.bias.data.cpu().numpy().tofile(fp)
    #                 # Load conv weights
    #                 conv_layer.weight.data.cpu().numpy().tofile(fp)
    #
    #     fp.close()

    def save_darknet_weights(self, path, include=None, ff=False):  # , discard_root=False):
        """
            @:param path    - path of the new weights file
            @:param cutoff  - save layers between 0 and cutoff (cutoff = -1 -> all are saved)
        """

        # cutoff needs to be: (a) a scalar, (b) a dict, or (c) None.
        if include is None:
            include = {}

        if isinstance(include, dict):
            for sx, defs_sx in self.sx_module_defs.items():
                if sx not in include:
                    include[sx] = np.arange(len(defs_sx))
        elif isinstance(include, list):
            assert len(self.sx_module_defs) == 1
            include = {'0': include}
        else:
            raise AttributeError

        modules_conv = []
        for sx, defs_sx in self.sx_module_defs.items():
            defs_sx = [module_def['type'] == 'convolutional'
                       for i, module_def in enumerate(defs_sx) if i in include[sx]]
            if sx != '0':
                modules_conv += [np.sum(defs_sx)]
            elif not ff:
                modules_conv += [np.sum(defs_sx)]
            else:  # ff
                modules_conv[-1] += np.sum(defs_sx)

        self.streams_len = np.array([len(modules_conv)] + modules_conv, dtype=np.int32)
        self.header_info[3] = self.seen

        fp = open(path, "wb")
        np.concatenate([self.header_info, self.streams_len]).tofile(fp)

        # if ff and len(self.sx_module_defs) > 1:
        #     root_defs, root_module_list = self.sx_module_defs['0'], self.sx_module_lists['0']
        #     del self.sx_module_defs['0']
        #     del self.sx_module_lists['0']

        for sx, defs_sx, modules_sx in common_items(self.sx_module_defs, self.sx_module_lists):
            defs_sx = [x for i, x in enumerate(defs_sx) if i in include[sx]]
            modules_sx = [x for i, x in enumerate(modules_sx) if i in include[sx]]
            if ff:
                if sx == '0':
                    continue
                else:
                    defs_sx += [x for i, x in enumerate(self.sx_module_defs['0']) if i in include[sx]]
                    modules_sx += [x for i, x in enumerate(self.sx_module_lists['0']) if i in include[sx]]

            for i, (module_def, module) in enumerate(zip(defs_sx, modules_sx)):
                if module_def["type"] == "convolutional":
                    conv_layer = module[0]
                    # If batch norm, load bn first
                    if module_def["batch_normalize"]:
                        bn_layer = module[1]
                        bn_layer.bias.data.cpu().numpy().tofile(fp)
                        bn_layer.weight.data.cpu().numpy().tofile(fp)
                        bn_layer.running_mean.data.cpu().numpy().tofile(fp)
                        bn_layer.running_var.data.cpu().numpy().tofile(fp)
                    # Load conv bias
                    else:
                        conv_layer.bias.data.cpu().numpy().tofile(fp)
                    # Load conv weights
                    conv_layer.weight.data.cpu().numpy().tofile(fp)

        fp.close()


def convert(cfg, input_weights, output_weights=None, cutoff=None, ff=False, include=None):
    # Converts between PyTorch and Darknet format per extension (i.e. *.weights convert to *.pt and vice versa)
    # from models import *; convert('cfg/yolov3-spp.cfg', 'weights/yolov3-spp.weights')

    # Initialize model
    model = Darknet(cfg)

    # Load weights and save
    if input_weights.endswith('.pth') or input_weights.endswith('.pt'):  # if PyTorch format
        if output_weights:
            assert output_weights.endswith('.weights')
        else:
            output_weights = 'converted.weights'

        model.load_state_dict(torch.load(input_weights, map_location='cpu')['model'])
        # model.save_darknet_weights(output_weights, cutoff=cutoff, ff=ff, save=save)
        os.makedirs(os.path.dirname(output_weights), exist_ok=True)
        model.save_darknet_weights(output_weights, include=rng2inds(include), ff=ff)
        print("Success: converted '%s' to '%s'" % (input_weights, output_weights))

    elif input_weights.endswith('.weights'):  # darknet format
        if output_weights:
            assert output_weights.endswith('.pth') or output_weights.endswith('.pt')
        else:
            output_weights = 'converted.pth'

        _ = model.load_darknet_weights(input_weights)
        chkpt = {'epoch': -1, 'best_loss': None, 'model': model.state_dict(), 'optimizer': None}
        torch.save(chkpt, output_weights)
        print("Success: converted '%s' to '%s'" % (input_weights, output_weights))

    else:
        print('Error: extension not supported.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_def", type=str, default="config/yolov3-1cls-dropout-supertiny.cfg", help="path to model definition file")
    parser.add_argument("--input_weights", type=str, help="if specified starts from checkpoint model")
    parser.add_argument("--output_weights", type=str, help="if specified starts from checkpoint model")
    parser.add_argument("--include", type=str, help="Discard (or keep) the 0-th stream (root) weights")
    parser.add_argument("--ff", action='store_true', help="Discard (or keep) the 0-th stream (root) weights")
    opt = parser.parse_args()
    print(opt)

    convert(opt.model_def, opt.input_weights, opt.output_weights, include=opt.include) #, discard_root=opt.discard_root)

