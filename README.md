# SE3CNN

The group SE(3) is the group of 3 dimensional rotations and translations.
This library aims to create SE(3) equivariant convolutional neural networks.

![](https://github.com/antigol/se3net/raw/master/examples/plots/kernels.png)

## Example
```python
import torch
from se3cnn import SE3Convolution

size = 32  # space size

scalar_field = torch.randn(1, 1, size, size, size)  # [batch, _, x, y, z]

Rs_in = [(1, 0)]  # 1 scalar field
Rs_out = [(1, 1)]  # 1 vector field
conv = SE3Convolution(Rs_in, Rs_out, size=5)
# conv.weight.size() == [2] (2 radial degrees of freedom)

vector_field = conv(scalar_field)  # [batch, vector component, x, y, z]

# vector_field.size() == [1, 3, 28, 28, 28]
```

## Hierarchy

- `se3cnn` contains the library
  - `se3cnn/convolution.py` defines `SE3Convolution` the main class of the library
  - `se3cnn/blocks` defines ways of introducing non linearity in an equivariant way
  - `se3cnn/batchnorm.py` equivariant batch normalization
  - `se3cnn/groupnorm.py` equivariant group normalization
  - `se3cnn/dropout.py` equivariant dropout
- `experiments` contains experiments made with the library
- `examples` simple scripts

## Dependencies

- [pytorch](https://pytorch.org)
- [lie_learn](https://github.com/AMLab-Amsterdam/lie_learn) is required to compute the irreducible representations of SO(3)
- scipy

## Usage

Install with
```
python setup.py install
```
