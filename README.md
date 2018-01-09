# SE3Net

The group SE(3) is the group of 3 dimensional rotations and translations.
This library aims to create SE(3) equivariant convolutional neural networks.

## Example

See file `arch/example.py`

## Hierarchy

- `se3_cnn` contains the library
  - `se3_cnn/convolution.py` defines `SE3Convolution` the main class of the library
  - `se3_cnn/blocks` defines `HighwayBlock` and `TensorProductBlock`, two ways of introducing non linearity in an equivariant way
  - `se3_cnn/batchnorm.py` equivariant batch normalization
  - `se3_cnn/dropout.py` equivariant dropout
- `arch` contains architectures made with the library
- `notebook` contains [jupyer](http://jupyter.org/) notebooks

## Dependencies

- [lie_learn](https://github.com/AMLab-Amsterdam/lie_learn) is required to compute the irreducible representations of SO(3)

## Usage

Install with
```
python setup.py install
```
