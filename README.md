# SE3Net

The group SE(3) is the group of 3 dimensional rotations and translations.
This library aims to create SE(3) equivariant convolutional neural networks.

![](https://github.com/antigol/se3net/raw/master/experiments/plots/kernels.png)

## Hierarchy

- `se3_cnn` contains the library
  - `se3_cnn/convolution.py` defines `SE3Convolution` the main class of the library
  - `se3_cnn/blocks` defines ways of introducing non linearity in an equivariant way
  - `se3_cnn/batchnorm.py` equivariant batch normalization
  - `se3_cnn/groupnorm.py` equivariant group normalization
  - `se3_cnn/dropout.py` equivariant dropout
- `experiments` contains experiments made with the library

## Dependencies

- [pytorch](https://pytorch.org) 
- [lie_learn](https://github.com/AMLab-Amsterdam/lie_learn) is required to compute the irreducible representations of SO(3)

## Usage

Install with
```
python setup.py install
```
