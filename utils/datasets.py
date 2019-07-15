import glob
import random
import os
import sys
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import cv2

from utils.preprocessing import image_normalization
from utils.augmentations import horisontal_flip, get_affine_transformation, apply_affine
from torch.utils.data import Dataset
import torchvision.transforms as transforms


def pad_to_square(img, pad_value):
    h, w = img.shape[:2]
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = cv2.copyMakeBorder(img, pad[2], pad[3], pad[0], pad[1], cv2.BORDER_CONSTANT, value=pad_value)  # padded square

    return img, pad


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


def random_resize(images, min_size=288, max_size=448):
    new_size = random.sample(list(range(min_size, max_size + 1, 32)), 1)[0]
    images = F.interpolate(images, size=new_size, mode="nearest")
    return images


class ImageFolder(Dataset):
    def __init__(self, folder_path, img_size=416):
        self.files = sorted(glob.glob("%s/*.*" % folder_path))
        self.img_size = img_size

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]
        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(img_path))
        # Pad to square resolution
        img, _ = pad_to_square(img, 0)
        # Resize
        img = resize(img, self.img_size)

        return img_path, img

    def __len__(self):
        return len(self.files)


class ListDataset(Dataset):
    def __init__(self, list_path, img_size=416, augment=True, multiscale=True, normalized_labels=True,
                 rescale_every_n_batches=10, img_norm=None, color_map=False):
        with open(list_path, "r") as file:
            self.img_files = file.readlines()

        self.label_files = [
            path.replace("images", "labels").replace(".png", ".txt").replace(".jpg", ".txt")
            for path in self.img_files
        ]
        self.img_size = img_size
        # self.max_objects = 100
        self.augment = augment
        self.multiscale = multiscale
        self.normalized_labels = normalized_labels
        self.rescale_every_n_batches = rescale_every_n_batches

        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0

        self.img_norm = None
        if img_norm:
            im_norm_split = img_norm.split(':')
            self.img_norm = im_norm_split[0]
            if len(im_norm_split) > 0:
                self.im_norm_rng = [float(x) for x in im_norm_split[1].split(',')]

        self.color_map = int(color_map)


    def __getitem__(self, index):
        return self.getitem(index)

    def getitem(self, index, Maff=None, flip_lr=None):
        """
        Returns index-th item out of dataset.

        :param index: Index indicating image to retrieve
        :param Maff: affine transformation matrix for data augmentation (rotates, translates, and scales image)
        :param flip_lr: boolean indicating whether to augment by flipping the retrieved image horizontally
        :return:
        """
        # ---------
        #  Image
        # ---------

        img_path = self.img_files[index % len(self.img_files)].rstrip()

        # Extract image as PyTorch tensor
        # img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))

        # Handle images with less than three channels
        # if len(img.shape) != 3:
        #     img = img.unsqueeze(0)
        #     img = img.expand((3, img.shape[1:]))

        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        h, w = img.shape[:2]

        h_factor, w_factor = (h, w) if self.normalized_labels else (1, 1)
        # Pad to square resolution
        img, pad = pad_to_square(img, 0)
        padded_h, padded_w = img.shape[:2]

        # ---------
        #  Label
        # ---------

        label_path = self.label_files[index % len(self.img_files)].rstrip()

        # targets = None
        # if os.path.exists(label_path):
        #     boxes = torch.from_numpy(np.loadtxt(label_path).reshape(-1, 5))
        #     # Extract coordinates for unpadded + unscaled image
        #     x1 = w_factor * (boxes[:, 1] - boxes[:, 3] / 2)
        #     y1 = h_factor * (boxes[:, 2] - boxes[:, 4] / 2)
        #     x2 = w_factor * (boxes[:, 1] + boxes[:, 3] / 2)
        #     y2 = h_factor * (boxes[:, 2] + boxes[:, 4] / 2)
        #     # Adjust for added padding
        #     x1 += pad[0]
        #     y1 += pad[2]
        #     x2 += pad[1]
        #     y2 += pad[3]
        #     # Returns (x, y, w, h)
        #     boxes[:, 1] = ((x1 + x2) / 2) / padded_w
        #     boxes[:, 2] = ((y1 + y2) / 2) / padded_h
        #     boxes[:, 3] *= w_factor / padded_w
        #     boxes[:, 4] *= h_factor / padded_h
        #
        #     targets = torch.zeros((len(boxes), 6))
        #     targets[:, 1:] = boxes

        labels_xyxy = None
        if os.path.exists(label_path):
            boxes = np.loadtxt(label_path).reshape(-1, 5)
            x1 = w_factor * (boxes[:, 1] - boxes[:, 3] / 2) + pad[0]
            y1 = h_factor * (boxes[:, 2] - boxes[:, 4] / 2) + pad[2]
            x2 = w_factor * (boxes[:, 1] + boxes[:, 3] / 2) + pad[1]
            y2 = h_factor * (boxes[:, 2] + boxes[:, 4] / 2) + pad[3]

            labels_xyxy = boxes.copy()
            labels_xyxy[:, 1] = x1
            labels_xyxy[:, 2] = y1
            labels_xyxy[:, 3] = x2
            labels_xyxy[:, 4] = y2

        if self.augment:
            if not Maff:
                Maff = get_affine_transformation(img.shape, degrees=(-5, 5), translate=(0.10, 0.10), scale=(0.90, 1.10), border=0)

            img, labels_xyxy = apply_affine(img, Maff, labels_xyxy, border=0)

        nl = len(labels_xyxy)
        if nl:
            labels_xywh = labels_xyxy.copy()
            labels_xywh[:, 1] = ((labels_xyxy[:, 1] + labels_xyxy[:, 3]) / 2) / padded_w
            labels_xywh[:, 2] = ((labels_xyxy[:, 2] + labels_xyxy[:, 4]) / 2) / padded_h
            labels_xywh[:, 3] = (labels_xyxy[:, 3] - labels_xyxy[:, 1]) / padded_w
            labels_xywh[:, 4] = (labels_xyxy[:, 4] - labels_xyxy[:, 2]) / padded_h

        if self.augment:
            if flip_lr or (flip_lr is None and random.random() > 0.5):
                img = np.fliplr(img)
                if nl:
                    labels_xywh[:, 1] = 1 - labels_xywh[:, 1]

        targets = None
        if nl:
            targets = torch.zeros((len(labels_xywh), 6))
            targets[:, 1:] = torch.from_numpy(labels_xywh)

        if self.img_norm:
            img = image_normalization(img, self.img_norm, self.im_norm_rng)

        if self.color_map:
            img = cv2.applyColorMap(img.astype(np.uint8), colormap=cv2.COLORMAP_JET)
        else:
            img = np.tile(img[:, :, np.newaxis], 3)

        # Normalize
        img = img[:, :, ::-1].transpose(2, 0, 1)  # Opencv's BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img, dtype=np.float32)  # uint8 to float32

        img /= 255.0  # 0 - 255 to 0.0 - 1.0

        return img_path, torch.from_numpy(img), targets

    def collate_fn(self, batch):
        paths, imgs, targets = list(zip(*batch))
        # Remove empty placeholder targets
        targets = [boxes for boxes in targets if boxes is not None]
        # Add sample index to targets
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i
        targets = torch.cat(targets, 0)
        # Selects new image size every tenth batch
        if self.multiscale and (self.batch_count % self.rescale_every_n_batches == 0):
            self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))
        # Resize images to input shape
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])
        self.batch_count += 1
        return paths, imgs, targets, self.img_size

    def __len__(self):
        return len(self.img_files)