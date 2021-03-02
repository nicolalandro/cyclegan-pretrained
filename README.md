[![Open on Torch Hub](https://img.shields.io/badge/Torch-Hub-red?logo=pytorch)]() 
[![License: MIT](https://img.shields.io/badge/license-MIT-lightgray)](LICENSE) 


[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]()

# Cyclegan pretrained
A pretrained cyclegan

|Original|Monet|Cezanne|Van Gogh|
|:---:|:---:|:---:|:---:|
|![original image](images/scala_madonnina_del_mare.jpeg)|![monet](images/monet.png)|![cezanne](images/cezanne.png)|![vangogh](images/vangogh.png)|

## How to use
```python
net = cyclegan(pretrained='cezanne', **{'topN': 6, 'device':'cpu', 'num_classes': 200})
```

## How to train

```bash
EPOCHS=200
DATASET="/path/to/datasets/cezanne2photo"
OUTPUT="/path/to/Models/cyclegan/cezanne"
GPU=1
CUDA_VISIBLE_DEVICES="${GPU}" python3 train.py --n_epochs="${EPOCHS}" --dataroot="${DATASET}" --output-folder="${OUTPUT}"
```