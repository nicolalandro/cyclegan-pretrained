#!/usr/bin/env bash

EPOCHS=200
DATASET="/home/super/datasets/monet2photo"
OUTPUT="/home/super/Models/cyclegan/monet"
GPU=1

mkdir -p $OUTPUT
CUDA_VISIBLE_DEVICES="${GPU}" python3 train.py --n_epochs="${EPOCHS}" --dataroot="${DATASET}" --output-folder="${OUTPUT}"
