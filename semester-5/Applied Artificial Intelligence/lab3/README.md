# Lab 3: Convolutional Neural Networks for Flower Classification

**Course:** Applied Artificial Intelligence (Semester 5, Fall 2026)
**Student:** Basioni Abdelsalam Ali Abdelsalam, group 465144

## Overview

This lab introduces convolutional neural networks (CNNs) for image classification using PyTorch and torchvision. The task is to classify photos of flowers into five classes: **daisy, dandelion, rose, sunflower, tulip**.

The notebook covers three parts:

1. **Transfer learning** with a pretrained ResNet-18, where only the final layer is trained.
2. **A custom CNN** designed to reduce the parameter count compared with the baseline network.
3. **A comparison** of three models: the baseline CNN from the template, the pretrained ResNet-18, and the custom CNN.

## Dataset

- Source: [Flowers Recognition on Kaggle](https://www.kaggle.com/alxmamaev/flowers-recognition)
- 4,327 images across 5 classes (dandelion 1052, tulip 994, rose 784, daisy 764, sunflower 733)
- Images are resized to 224×224 and normalized with ImageNet mean and standard deviation.
- Split: 3,327 images for training and 1,000 for validation (random split with seed 0).

The dataset is not included in this repository. Download it and place it as:

```
data/flowers/
├── daisy/
├── dandelion/
├── rose/
├── sunflower/
└── tulip/
```

On Google Colab, the notebook expects the folder at `/content/drive/My Drive/data/flowers`.

## Requirements

- Python 3.8+
- PyTorch and torchvision (CUDA is recommended but not required)
- numpy, pandas, matplotlib, tqdm


## How to Run

1. Put the dataset in `data/flowers/` in your google drive.
2. Open the notebook in Jupyter or Google Colab. On Colab, enable a GPU under *Runtime → Change runtime type*.
3. Run all cells from top to bottom. Random seeds are fixed (`random.seed(0)`, `torch.manual_seed(0)`) so the splits are reproducible.

## Models and Results

| Model | Trainable params | Epochs | Train acc. | Val acc. |
|---|---:|---:|---:|---:|
| Baseline CNN (template) | 207,175,365 | 3 | 48.1% | 48.7% |
| ResNet-18, transfer learning | 2,565 | 5 | 87.7% | **87.9%** |
| Custom CNN | 1,048,613 | 10 | 88.7% | 70.1% |

**Key observations**

- **ResNet-18 with transfer learning** performed best. Only the new final linear layer (512 → 5) was trained, and the frozen ImageNet features did most of the work.
- **The baseline CNN** has 207 million parameters, and about 205 million of them sit in the first fully connected layer (256·28·28 → 1024). This makes it slow to train and prone to overfitting, and it reached only 48.7% after 3 epochs.
- **The custom CNN** shrinks the feature map to 7×7 with five pooling stages, which cuts the parameters to about 1 million. It reached 70.9% at its best validation epoch (epoch 8), then overfit: training accuracy rose to 88.7% while validation accuracy fell.
