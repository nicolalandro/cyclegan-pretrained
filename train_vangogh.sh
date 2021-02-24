#!/usr/bin/env bash

EPOCHS=200
DATASET="/home/super/Datasets/vangogh2photo"
OUTPUT="/home/super/Models/cyclegan/vangog"
GPU=0

mkdir -r -p $OUTPUT
CUDA_VISIBLE_DEVICES="${GPU}" python3 train.py --n_epochs="${EPOCHS}" --dataroot="${DATASET}" --output-folder="${OUTPUT}"
