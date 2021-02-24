#!/usr/bin/env bash

EPOCHS=1

# DATASET="/media/mint/Barracuda/Datasets/CycleGanDatasets/vangogh2photo"
# OUTPUT="/media/mint/Barracuda/Models/cyclegan/vangog"

# python3 train.py --n_epochs="${EPOCHS}" --dataroot="${DATASET}" --output-folder="${OUTPUT}"

DATASET="/media/mint/Barracuda/Datasets/CycleGanDatasets/cezanne2photo"
OUTPUT="/media/mint/Barracuda/Models/cyclegan/cezanne"

python3 train.py --n_epochs="${EPOCHS}" --dataroot="${DATASET}" --output-folder="${OUTPUT}"

DATASET="/media/mint/Barracuda/Datasets/CycleGanDatasets/monet2photo"
OUTPUT="/media/mint/Barracuda/Models/cyclegan/monet"

python3 train.py --n_epochs="${EPOCHS}" --dataroot="${DATASET}" --output-folder="${OUTPUT}"

DATASET="/media/mint/Barracuda/Datasets/CycleGanDatasets/ukiyoe2photo"
OUTPUT="/media/mint/Barracuda/Models/cyclegan/ukiyoe"

python3 train.py --n_epochs="${EPOCHS}" --dataroot="${DATASET}" --output-folder="${OUTPUT}"

DATASET="/media/mint/Barracuda/Datasets/CycleGanDatasets/maps"
OUTPUT="/media/mint/Barracuda/Models/cyclegan/maps"

python3 train.py --n_epochs="${EPOCHS}" --dataroot="${DATASET}" --output-folder="${OUTPUT}"