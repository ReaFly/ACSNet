#  Adaptive Context Selection for Polyp Segmentation

##  Introduction

This repository contains the PyTorch implementation of:

Adaptive Context Selection for Polyp Segmentation, MICCAI 2020.

##  Requirements

* torch
* torchvision 
* tqdm
* opencv
* scipy
* skimage
* PIL
* numpy

##  Usage

####  1. Training

```bash
python train.py  --mode train  --dataset kvasir_SEG  
--train_data_dir /path-to-train_data  --valid_data_dir  /path-to-valid_data
```



####  2. Inference

```bash
python test.py  --mode test  --load_ckpt checkpoint 
--dataset kvasir_SEG    --test_data_dir  /path-to-test_data
```



##  Citation

If you feel this work is helpful, please cite our paper

```
@inproceedings{zhang2020adaptive,
  title={Adaptive Context Selection for Polyp Segmentation},
  author={Zhang, Ruifei and Li, Guanbin and Li, Zhen and Cui, Shuguang and Qian, Dahong and Yu, Yizhou},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={253--262},
  year={2020},
  organization={Springer}
} 
```





