#!/usr/bin/env bash

EPOCHS=200
DATASET="/home/super/datasets/cezanne2photo"
OUTPUT="/home/super/Models/cyclegan/cezanne"
GPU=2

mkdir -p $OUTPUT
CUDA_VISIBLE_DEVICES="${GPU}" python3 train.py --n_epochs="${EPOCHS}" --dataroot="${DATASET}" --output-folder="${OUTPUT}"
