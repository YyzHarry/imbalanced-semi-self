# Rethinking the Value of Labels for Improving Class-Imbalanced Learning

This repository contains the implementation code for paper: <br>
__Rethinking the Value of Labels for Improving Class-Imbalanced Learning__ <br>
[Yuzhe Yang](http://www.mit.edu/~yuzhe/), and [Zhi Xu](http://www.mit.edu/~zhixu/) <br>
_34th Conference on Neural Information Processing Systems (__NeurIPS 2020__)_ <br>
[[Website](http://www.mit.edu/~yuzhe/imbalanced-semi-self.html)] [[arXiv](https://arxiv.org/abs/2006.07529)] [[Paper](https://arxiv.org/pdf/2006.07529.pdf)] [[Slides]()] [[Video]()]


## Overview
In this work, we show theoretically and empirically that, both __semi-supervised learning__ (using unlabeled data) and __self-supervised pre-training__ (first pre-train the model with self-supervision) can substantially improve the performance on imbalanced (long-tailed) datasets, regardless of the imbalanceness on labeled/unlabeled data and the base training techniques.

__Semi-Supervised Imbalanced Learning__: 
Using unlabeled data helps to shape clearer class boundaries and results in better class separation, especially for the tail classes.
![semi](assets/tsne_semi.png)

__Self-Supervised Imbalanced Learning__:
Self-supervised pre-training (SSP) helps mitigate the tail classes leakage during testing, which results in better learned boundaries and representations.
![self](assets/tsne_self.png)


## Installation

### Prerequisites
- Download [CIFAR](https://www.cs.toronto.edu/~kriz/cifar.html) & [SVHN](http://ufldl.stanford.edu/housenumbers/) dataset, and place them in your `data_path`
- Download [ImageNet](http://image-net.org/download) & [iNaturalist 2018](https://github.com/visipedia/inat_comp/tree/master/2018) dataset, and place them in your `data_path`
- Change the `data_root` in [`imagenet_inat/main.py`](./imagenet_inat/main.py) accordingly for ImageNet-LT & iNaturalist 2018

### Dependencies
- PyTorch (>= 1.2, tested on 1.4)
- yaml
- scikit-learn
- TensorboardX


## Code Overview

### Main Files
- [`train_semi.py`](train_semi.py): train model with extra unlabeled data, on CIFAR-LT / SVHN-LT
- [`train.py`](train.py): train model with (or without) SSP, on CIFAR-LT / SVHN-LT
- [`imagenet_inat/main.py`](./imagenet_inat/main.py): train model with (or without) SSP, on ImageNet-LT / iNaturalist 2018
- [`pretrain_rot.py`](pretrain_rot.py) & [`pretrain_moco.py`](pretrain_moco.py): self-supervised pre-training using [Rotation prediction](https://arxiv.org/pdf/1803.07728.pdf) or [MoCo](https://arxiv.org/abs/1911.05722)

### Main Arguments
- `--dataset`: name of chosen long-tailed dataset
- `--imb_factor`: imbalance factor (inverse value of imbalance ratio `\rho` in the paper)
- `--imb_factor_unlabel`: imbalance factor for unlabeled data (inverse value of unlabeled imbalance ratio `\rho_U`)
- `--pretrained_model`: path to self-supervised pre-trained models
- `--resume`: path to resume checkpoint (also for evaluation)


## Getting Started

### Semi-Supervised Imbalanced Learning

#### Unlabeled data sourcing

__CIFAR-10-LT__: CIFAR-10 unlabeled data is prepared following [this repo](https://github.com/yaircarmon/semisup-adv) using the [80M TinyImages](https://people.csail.mit.edu/torralba/publications/80millionImages.pdf). In short, a data sourcing model is trained to distinguish CIFAR-10 classes and an "non-CIFAR" class. For each class, images are then ranked based on the prediction confidence, and unlabeled (imbalanced) datasets are constructed accordingly. Use the following link to download the prepared unlabeled data, and place in your `data_path`:
- [Unlabeled dataset for CIFAR-10-LT from TinyImages](https://drive.google.com/file/d/1SODQBUvv2qycDivBb4nhHaCk3TMzaVM4/view?usp=sharing)

__SVHN-LT__: Since its own dataset contains an extra part with 531.1K additional (labeled) samples, they are directly used to simulate the unlabeled dataset.

#### Semi-supervised learning with pseudo-labeling

To perform pseudo-labeling (self-training), first a base classifier is trained on original imbalanced dataset. With the trained base classifier, pseudo-labels can be generated using
```bash
python gen_pseudolabels.py --resume <ckpt-path> --data_dir <data_path> --output_dir <output_path> --output_filename <save_name>
```
We provide generated pseudo label files for CIFAR-10-LT & SVHN-LT with `\rho=50`, using base model with standard CE training:
- [Generated pseudo labels for CIFAR-10-LT with `\rho=50`](https://drive.google.com/file/d/1Z4rwaqzjNoNQ27sofx1aDl8OLH-etoyP/view?usp=sharing)
- [Generated pseudo labels for SVHN-LT with `\rho=50`](https://drive.google.com/file/d/19VeMQ07unVq3hIjLN5LiXWZNTI4CiN5F/view?usp=sharing)

To train with unlabeled data, for example, on CIFAR-10-LT with `\rho=50` and `\rho_U=50`
```bash
python train_semi.py --dataset cifar10 --imb_factor 0.02 --imb_factor_unlabel 0.02
```

### Self-Supervised Imbalanced Learning

#### Self-supervised pre-training (SSP)
To perform Rotation SSP on CIFAR-10-LT with `\rho=100`
```bash
python pretrain_rot.py --dataset cifar10 --imb_factor 0.01
```

To perform MoCo SSP on ImageNet-LT
```bash
python pretrain_moco.py --dataset imagenet --data <data_path>
```

#### Network training with SSP models
Train on CIFAR-10-LT with `\rho=100`
```bash
python train.py --dataset cifar10 --imb_factor 0.01 --pretrained_model <path_to_ssp_model>
```

Train on ImageNet-LT / iNaturalist 2018 (change `model_dir` to SSP model path in the config file)
```bash
python -m imagenet_inat.main --cfg <path_to_ssp_config>
```


## Results and Models



## Acknowledgements
This code is partly based on the open-source implementations from the following sources:
[OpenLongTailRecognition](https://github.com/zhmiao/OpenLongTailRecognition-OLTR), [classifier-balancing](https://github.com/facebookresearch/classifier-balancing), [LDAM-DRW](https://github.com/kaidic/LDAM-DRW), [MoCo](https://github.com/facebookresearch/moco), and [semisup-adv](https://github.com/yaircarmon/semisup-adv).


## Citation
If you find the idea or code useful for your research, please cite [our paper](https://arxiv.org/abs/2006.07529):
```bib
@inproceedings{yang2020rethinking,
  title={Rethinking the Value of Labels for Improving Class-Imbalanced Learning},
  author={Yang, Yuzhe and Xu, Zhi},
  booktitle={Conference on Neural Information Processing Systems (NeurIPS)},
  year={2020},
}
```
