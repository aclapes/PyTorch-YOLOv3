from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import parse_data_config

import os
import time
import argparse

import torch
from torch.utils.data import DataLoader

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_image_folder", type=str, default="data/samples", help="path to dataset")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
    parser.add_argument('--data_config', type=str, default='data/coco.data', help='coco.data file path')
    parser.add_argument("--conf_thres", type=float, default=0.5, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.20, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=8, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=1, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--scale_factor", type=str, default="0.25,2", help="scale factor of images for visualization only")
    parser.add_argument("--detection_output", type=str, default="detection_images/", help="where to output detections")
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
    data_configs = [parse_data_config(cfg) for cfg in opt.data_config.split(',')]

    # classes = load_classes(data_config['names'])  # Extracts class labels from file
    classes = None
    for cfg in data_configs:
        classes_i = load_classes(cfg['names'])
        if classes:
           assert classes_i == classes
        classes = classes_i

    datasets = [ImageFolder(input, img_norm=cfg['normalization'], color_map=cfg['color_map'])
                for input, cfg in zip(opt.input_image_folder.split(','), data_configs)]

    multidataset = ParallelImageFolder(datasets, img_size=opt.img_size)

    dataloader = torch.utils.data.DataLoader(
        multidataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.n_cpu,
        pin_memory=True,
        collate_fn=multidataset.collate_fn,
    )

    detection_output = opt.detection_output.split(',')
    for path in detection_output:
        os.makedirs(path, exist_ok=True)

    scale_factor = [float(sf) for sf in opt.scale_factor.split(',')]

    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(classes))]

    st_time = time.time()
    for batch_i, collation in enumerate(dataloader):
        paths, imgs, imgs0 = list(zip(*collation))
        # Input batch contains several images
        input_tensor = [imgs_i.type(Tensor) for imgs_i in imgs]

        # Get detections
        with torch.no_grad():
            dets = model(input_tensor)
            dets = non_max_suppression(dets, opt.conf_thres, opt.nms_thres)

        # For each image in the batch process its detections
        for i, (paths_i, imgs0_i, dets_i) in enumerate(zip(zip(*paths), zip(*imgs0), dets)):
            print("[%d/%d] Image(s): '%s'" % (batch_i * opt.batch_size + i + 1, len(multidataset), ", ".join(paths_i)))

            for j, path in enumerate(paths_i):
                h, w = imgs0_i[j].shape[:2]
                img0 = cv2.resize(imgs0_i[j],
                                  (int(w*scale_factor[j]), int(h*scale_factor[j])),
                                  interpolation=cv2.INTER_AREA)

                # Draw bounding boxes and labels of detections
                if dets_i is not None:
                    # Rescale boxes to original image
                    dets_i_rsc_j = rescale_boxes(dets_i, opt.img_size, img0.shape[:2])

                    n = len(dets_i_rsc_j)
                    if j < 1: print('Detections:', end=' ')
                    # Draw bounding boxes and labels of detections
                    for k, (x1, y1, x2, y2, conf, cls_conf, cls) in enumerate(dets_i_rsc_j):
                        # Add bbox to the image
                        label = '%s (%.2f)' % (classes[int(cls)], conf)
                        if j < 1: print('[%d/%d] %s' % (k+1, n, label), end=(', ' if k < len(dets_i_rsc_j) - 1 else '.\n'))
                        plot_one_box(x1, y1, x2, y2, img0, label=label, color=colors[int(cls)])

                filename = paths_i[j].split("/")[-1].split(".")[0]
                cv2.imwrite(f"{detection_output[j]}/{filename}.jpg", img0)

    print("Done. (%2.2f)" % (time.time() - st_time))