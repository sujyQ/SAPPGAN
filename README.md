# SAPPGAN

Pytorch implementation of "Semantic-Aware Face Deblurring with Pixel-Wise Projection Discriminator", IEEE Access, 2023

[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10036438)

## Overview
![framework](./figures/framework.PNG)

## Requirements
+ Python
+ PyTorch
+ numpy
+ torchvision
+ PIL
+ scipy
+ tqdm


## Train
### 1. Prepare training data
1.1 

### 2. Begin to train

## Test
### 1. Prepare test data
### 2. Begin to test

## Comparative Results
### Quantitative Comparisons on MSPL test set
![framework](./figures/mspl-testset.PNG)

### Qualitative Comparisons on MSPL-Center test set
![framework](./figures/mspl-center.PNG)

### Qualitative Comparisons on MSPL-Random test set
![framework](./figures/mspl-random.PNG)

### Real Blur
![framework](./figures/realblur.PNG)


## Citation
```
@article{sujy2023semantic,
  title={Semantic-Aware Face Deblurring with Pixel-Wise Projection Discriminator},
  author={Sujy, HAN and Lee, Tae Bok and Heo, Yong Seok},
  journal={IEEE Access},
  year={2023},
  publisher={IEEE}
}
```

## Acknowledgements
This code is built on [U-Net GAN](https://github.com/boschresearch/unetgan) and [MSPLGAN](https://github.com/dolphin0104/MSPL-GAN). We thank the authors for sharing the codes.
