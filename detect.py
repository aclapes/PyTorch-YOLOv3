from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import parse_data_config

import os
import sys
import time
import datetime
import argparse

from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_image_folder", type=str, default="data/samples", help="path to dataset")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
    parser.add_argument('--data_config', type=str, default='data/coco.data', help='coco.data file path')
    parser.add_argument("--conf_thres", type=float, default=0.5, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.20, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=8, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--scale_factor", type=float, default=1, help="scale factor of images for visualization only")
    parser.add_argument("--detection_images", type=str, default="detection_images/", help="where to output detections")
    opt = parser.parse_args()
    print(opt)

    # Set CUDA-related variables
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    # Set up model
    model = Darknet(opt.model_def).to(device)

    if opt.weights_path.endswith(".pth"):
        model.load_state_dict(torch.load(opt.weights_path, map_location=device)['model'])
    elif opt.weights_path.endswith(".weights"):
        model.load_darknet_weights(opt.weights_path)

    model.eval()  # Set in evaluation mode

    # Prepare data
    data_config = parse_data_config(opt.data_config)
    classes = load_classes(data_config['names'])  # Extracts class labels from file

    dataset = ImageFolder(opt.input_image_folder, img_size=opt.img_size,
                    img_norm=data_config['normalization'], color_map=data_config['color_map'])

    dataloader = DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.n_cpu,
        collate_fn=dataset.collate_fn
    )

    os.makedirs(f"{opt.detection_images}", exist_ok=True)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(classes))]

    st_time = time.time()
    for batch_i, (paths, imgs, imgs0) in enumerate(dataloader):
        # Input batch contains several images
        input_tensor = imgs.type(Tensor)

        # Get detections
        with torch.no_grad():
            dets = model(input_tensor)
            dets = non_max_suppression(dets, opt.conf_thres, opt.nms_thres)

        # For each image in the batch process its detections
        for i, (path, img0, dets_img) in enumerate(zip(paths, imgs0, dets)):
            print("[%d/%d] Image: '%s'" % (batch_i * opt.batch_size + i + 1, len(dataset), path))

            img0 = cv2.resize(img0, (int(img0.shape[0] * opt.scale_factor), int(img0.shape[1] * opt.scale_factor)),
                              interpolation=cv2.INTER_AREA)

            # Draw bounding boxes and labels of detections
            if dets_img is not None:
                # Rescale boxes to original image
                dets_img = rescale_boxes(dets_img, opt.img_size, img0.shape[:2])

                n = len(dets_img)
                print('Detections', end=': ')
                # Draw bounding boxes and labels of detections
                for k, (x1, y1, x2, y2, conf, cls_conf, cls) in enumerate(dets_img):
                    # Add bbox to the image
                    label = '%s (%.2f)' % (classes[int(cls)], conf)
                    print('[%d/%d] %s' % (k+1, n, label), end=(', ' if k < len(dets_img) - 1 else '.\n'))
                    plot_one_box(x1, y1, x2, y2, img0, label=label, color=colors[int(cls)])

            filename = path.split("/")[-1].split(".")[0]
            cv2.imwrite(f"{opt.detection_images}/{filename}.png", img0)

    print("Done. (%2.2f)" % (time.time() - st_time))