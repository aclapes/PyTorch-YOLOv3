import cv2
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_fusion_dir", type=str,
                        default="/home/aclapes/Downloads/detection-lami.2/2019-05-07_14.54.16",
                        help="size of each image batch")
    parser.add_argument("--input_modality_dirs", type=str,
                        default="/home/aclapes/Downloads/detection-2/2019-05-07_14.54.16,/home/aclapes/Downloads/detection-7/2019-05-07_14.54.16",
                        help="size of each image batch")
    parser.add_argument("--prefixes", type=str,
                        default="rs/depth-floor-1/,pt/thermal/",
                        help="size of each image batch")
    parser.add_argument("--fps", type=float, default=6, help="size of each image dimension")
    parser.add_argument("--img_size", type=int, default=320, help="size of each image dimension")

    parser.add_argument("--output", type=str, default="/home/aclapes/2019-05-07_14.54.16-new.mp4", help="path to data config file")

    opt = parser.parse_args()
    print(opt)

    prefixes = opt.prefixes.split(',')
    parents_modalities = [dir.rstrip('/') for dir in opt.input_modality_dirs.split(',')]
    parent_fusion = opt.input_fusion_dir
    seq_dir = os.path.basename(opt.input_fusion_dir.rstrip('/'))

    files = []
    for parent, pfx in zip(parents_modalities, prefixes):
        fusion_pfx_files = sorted(glob.glob(os.path.join(parent_fusion, pfx, "*.*")))
        pfx_files = [os.path.join(parent, ff.split(f"{seq_dir}/")[-1]) for ff in fusion_pfx_files]
        files += [list(zip(fusion_pfx_files, pfx_files))]

    writer = None
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    pbar = tqdm(enumerate(zip(*files)), total=len(files))
    for k, frame_pairs in pbar:
        img_pairs = []
        img_msc = None
        for j, pair in enumerate(frame_pairs):
            # img_pair = [cv2.imread(path, cv2.IMREAD_UNCHANGED) for path in pair]
            for i, path in enumerate(pair):
                im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                im, _ = pad_to_square(im, 0)
                im = cv2.resize(im, (opt.img_size, opt.img_size))
                h, w = im.shape[:2]
                if img_msc is None:
                    img_msc = np.empty((opt.img_size*2, opt.img_size*len(prefixes), im.shape[2]), dtype=np.uint8)
                    if writer is None:
                        writer = cv2.VideoWriter(opt.output, fourcc, opt.fps, img_msc.shape[:2])

                x1, y1, x2, y2 = j*w, i*h, (j+1)*w, (i+1)*h
                img_msc[y1:y2, x1:x2, :] = im

        writer.write(img_msc)
    writer.release()

    quit()