from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *

import os
import sys
import time
import datetime
import argparse
import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim

from terminaltables import AsciiTable


def evaluate(model, data_configs, output, iou_thres, conf_thres, nms_thres, img_size, batch_size, num_workers=8):
    model.eval()

    # Get dataloader
    datasets = [ListDataset(cfg["valid"], img_norm=cfg['normalization'], color_map=cfg['color_map'])
                for cfg in data_configs]

    multidataset = ParallelDataset(datasets,
                                   img_size=img_size,
                                   augment=False,
                                   multiscale=False)

    dataloader = torch.utils.data.DataLoader(
        multidataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=multidataset.collate_fn
    )

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    loss_batches_eval = []
    # for batch_i, (_, imgs, targets, _) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):
    for batch_i, (batch_data, img_size) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):
        paths, imgs, targets = list(zip(*batch_data))

        imgs = [Variable(x.type(Tensor)) for x in imgs]
        targets = Variable(targets[0].type(Tensor), requires_grad=False)

        if batch_i == 0:
            for k, imgs_k in enumerate(imgs):
                plot_images(imgs=imgs_k, targets=targets,
                            fname=os.path.join(output, 'test_batch-%g.%g.jpg') % (batch_i, k))

        # imgs = Variable(imgs.type(Tensor), requires_grad=False)
        with torch.no_grad():
            loss, _, outputs = model(imgs, targets) # targets only evaluation loss
            outputs = non_max_suppression(outputs, conf_thres=conf_thres, nms_thres=nms_thres)

        # Extract labels
        labels += targets[:, 1].tolist()
        # Rescale target
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        targets[:, 2:] *= img_size

        sample_metrics += get_batch_statistics(outputs, targets.cpu(), iou_threshold=iou_thres)
        loss_batches_eval += [loss.item()]

    # Concatenate sample statistics
    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)

    return loss_batches_eval, precision, recall, AP, f1, ap_class


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8, help="size of each image batch")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="config/coco.data", help="path to data config file")
    parser.add_argument("--pretrained_weights", type=str, default="weights/yolov3.weights", help="path to weights file")
    # parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="iou threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.5, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.5, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--output", type=str, default="output/", help="path to data config file")

    opt = parser.parse_args()
    print(opt)

    os.makedirs(opt.output, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_configs = [parse_data_config(cfg_file) for cfg_file in opt.data_config.split(',')]

    # data_config = parse_data_config(opt.data_config)
    # valid_path = data_config["valid"]
    # class_names = load_classes(data_config["names"])

    # Initiate model
    model = Darknet(opt.model_def).to(device)
    # if opt.weights_path.endswith(".weights"):
    #     # Load darknet weights
    #     model.load_darknet_weights(opt.weights_path)
    # else:
    #     # Load checkpoint weights
    #     model.load_state_dict(torch.load(opt.weights_path))
    weights = [w for w in opt.pretrained_weights.split(',')]
    if len(weights) == 1 and (weights[0].endswith('.pth') or weights[0].endswith('.pt')):
        chkpt = torch.load(weights[0], map_location=device)  # load checkpoint
        model.load_state_dict(chkpt['model'])
        del chkpt
    else:
        for k, w in enumerate(weights):
            _ = model.load_darknet_weights(w, k=k, cutoff=-1)

    print("Compute mAP...")

    loss_batches_eval, precision, recall, AP, f1, ap_class = evaluate(
        model,
        data_configs,
        opt.output,
        iou_thres=opt.iou_thres,
        conf_thres=opt.conf_thres,
        nms_thres=opt.nms_thres,
        img_size=opt.img_size,
        batch_size=8,
    )

    evaluation_metrics = [
        ("val_loss", np.mean(loss_batches_eval)),
        ("val_precision", precision.mean()),
        ("val_recall", recall.mean()),
        ("val_mAP", AP.mean()),
        ("val_f1", f1.mean()),
    ]

    summary_table = [[metric_name for metric_name, _ in evaluation_metrics]]
    summary_table += [["%.5f" % metric_val for _, metric_val in evaluation_metrics]]
    print(AsciiTable(summary_table).table)

    # print("Average Precisions:")
    # for i, c in enumerate(ap_class):
    #     print(f"+ Class '{c}' ({class_names[c]}) - AP: {AP[i]}")
    #
    # print(f"mAP: {AP.mean()}")
