#!/bin/bash

python3 utils/detection_to_vid.py --input_fusion_dir=/home/aclapes/Downloads/detection-lami.2/2019-05-07_14.54.16 \
        --input_modality_dirs=/home/aclapes/Downloads/detection-2/2019-05-07_14.54.16,/home/aclapes/Downloads/detection-7/2019-05-07_14.54.16 \
        --output=/home/aclapes/2019-05-07_14.54.16.mp4

python3 utils/detection_to_vid.py --input_fusion_dir=/home/aclapes/Downloads/detection-lami.2/2019-05-09_14.46.25 \
        --input_modality_dirs=/home/aclapes/Downloads/detection-2/2019-05-09_14.46.25,/home/aclapes/Downloads/detection-7/2019-05-09_14.46.25 \
        --output=/home/aclapes/2019-05-09_14.46.25.mp4

python3 utils/detection_to_vid.py --input_fusion_dir=/home/aclapes/Downloads/detection-lami.2/2019-05-09_14.59.57 \
        --input_modality_dirs=/home/aclapes/Downloads/detection-2/2019-05-09_14.59.57,/home/aclapes/Downloads/detection-7/2019-05-09_14.59.57 \
        --output=/home/aclapes/2019-05-09_14.59.57.mp4

python3 utils/detection_to_vid.py --input_fusion_dir=/home/aclapes/Downloads/detection-lami.2/2019-05-09_19.18.22 \
        --input_modality_dirs=/home/aclapes/Downloads/detection-2/2019-05-09_19.18.22,/home/aclapes/Downloads/detection-7/2019-05-09_19.18.22 \
        --output=/home/aclapes/2019-05-09_19.18.22.mp4