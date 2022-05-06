# Semi-Supervised Learning Methods

This repository is unofficial implementation of following papers with Tensorflow 2.0. The corresponding folder name is written in parenthesis.

- Variational AutoEncoder:
  - [Semi-supervised Learning with Deep Generative Models](https://proceedings.neurips.cc/paper/2014/hash/d523773c6b194f37b938d340d5d02232-Abstract.html) (`dgm`)
  - [Auxiliary deep generative models](http://proceedings.mlr.press/v48/maaloe16.html) (`adgm`)
  - [Ladder variational autoencoders](https://proceedings.neurips.cc/paper/2016/file/6ae07dcb33ec3b7c814df797cbda0f87-Paper.pdf) (`ladder`)
  - [Semi-supervised disentanglement of class-related and class-independent factors in vae](https://arxiv.org/pdf/2102.00892.pdf) (`partedvae`)
  - [SHOT-VAE: semi-supervised deep generative models with label-aware ELBO approximations](https://www.aaai.org/AAAI21Papers/AAAI-260.FengHZ.pdf) (`shotvae`)

- Classification models:
  - [Temporal ensembling for semi-supervised learning](https://arxiv.org/pdf/1610.02242.pdf?ref=https://githubhelp.com) (`pi`)
  - [Virtual adversarial training: a regularization method for supervised and semi-supervised learning](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8417973) (`vat`)
  - [Label propagation for deep semi-supervised learning](https://openaccess.thecvf.com/content_CVPR_2019/papers/Iscen_Label_Propagation_for_Deep_Semi-Supervised_Learning_CVPR_2019_paper.pdf) (`lp`)
  - [Mixmatch: A holistic approach to semi-supervised learning](https://proceedings.neurips.cc/paper/2019/file/1cd138d0499a68f4bb72bee04bbec2d7-Paper.pdf) (`mixmatch`)
  - [Pseudo-labeling and confirmation bias in deep semi-supervised learning](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9207304) (`plcb`)

## Package Dependencies

```setup
python==3.7
numpy==1.19.5
tensorflow==2.4.0
```
Additional package requirements for this repository are described in `requirements.txt`.

## How to Training & Evaluation  

`labeled_examples` is the number of labeled datsets for running and we provide configuration `.yaml` files for 100 labeled datsets of MNIST and 4000 labeled datasets of CIFAR-10. And we add required tests and evaluations at the end of code.

1. MNIST dataset running

```
python mnist/main.py --config_path "configs/mnist_{labeled_examples}.yaml"
```   

2. CIFAR-10 dataset running

```
python main.py --config_path "configs/cifar10_{labeled_examples}.yaml"
```   

## Results (CIFAR-10)

The number in parenthesis next to the name of model is the number of parameters in classifier. Inception score of classification model is not computed.

|       Model      | Classification error | Inception Score |
|:----------------:|:--------------------:|:---------------:|
| Pi-model(4.5M)   |               17.58% |               - |
| VAT(4.5M)        |                13.70 |               - |
| MixMatch(5.8M)   |                5.55% |               - |
| PLCB(4.5M)       |                7.69% |               - |
| M2(4.5M)         |               27.69% |     1.85 (0.05) |
| Parted-VAE(5.8M) |               31.85% |      1.58(0.04) |
| SHOT-VAE(5.8M)   |                5.91% |     3.46 (0.18) |

## Reference codes

- https://github.com/wohlert/semi-supervised-pytorch
- https://github.com/sinahmr/parted-vae
- https://github.com/FengHZ/AAAI2021-260
- https://github.com/hiram64/temporal-ensembling-semi-supervised/tree/master/lib
- https://github.com/takerum/vat_tf/tree/c5125d267531ce0f10b2238cf95604d287de63c8
- https://github.com/9310gaurav/virtual-adversarial-training
- https://github.com/ahmetius/LP-DeepSSL
- https://github.com/ntozer/mixmatch-tensorflow2.0
- https://github.com/EricArazo/PseudoLabeling
