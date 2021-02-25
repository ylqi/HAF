# HIERARCHICAL ATTENTION FUSION FOR GEO-LOCALIZATION 

<p align="center">
    <img src="figs/framework.png" width="80%">
</p>

## Introduction

`HAF` extracts the multi-scale feature maps from a convolutional neural network (CNN) to perform hierarchical attention fusion for image representations. This repo is the PyTorch implementation of ICASSP2021 paper "HIERARCHICAL ATTENTION FUSION FOR GEO-LOCALIZATION"
 [[pdf](https://arxiv.org/abs/2102.09186)] 

## Installation

Please find detailed steps [Here](docs/INSTALL.md) for installation and dataset preparation.

## Train & Test

Please find details [Here](docs/REPRODUCTION.md) for step-by-step instructions.

## Model Zoo

Please refer to [Here](docs/MODEL_ZOO.md) for trained models.

## Inference on a single image 

Please refer to [Here](docs/INFERENCE.md) for inference on a single image.

## Train on customized dataset
Please refer to [Here](docs/INSTALL.md#use-custom-dataset-optional) to prepare your own dataset.

## License

`HAF` is released under the MIT license.


## Citation

If you find this repo useful for your research, please consider citing the paper
```
@inproceedings{yan2021densernet,
    title={Hierarchical Attention Fusion for Geo-localization},
    author={Liqi Yan, Yiming Cui, Yingjie Chen, Dongfang Liu},
    booktitle={ICASSP}
    year={2021},
}
```

## Acknowledgements
We truely thanksful of the following piror efforts in terms of knowledge contributions and open-source repos. Particularly, "ASLFeat" has a similar approach to ours but using strong supervision.
+ NetVLAD: CNN architecture for weakly supervised place recognition (CVPR'16) [[paper]](https://arxiv.org/abs/1511.07247) [[official code (pytorch-NetVlad)]](https://github.com/Nanne/pytorch-NetVlad)
+ SARE: Stochastic Attraction-Repulsion Embedding for Large Scale Image Localization (ICCV'19) [[paper]](https://arxiv.org/abs/1808.08779) [[official code (MatConvNet)]](https://github.com/Liumouliu/deepIBL) 
+ ASLFeat: Learning Local Features of Accurate Shape and Localization (CVPR'20) [[paper]](https://arxiv.org/abs/2003.10071) [[official code]](https://github.com/lzx551402/ASLFeat)