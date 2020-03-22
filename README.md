# E3NN
The group E(3) is the group of 3 dimensional rotations, translations and mirror.
This library aims to create E(3) equivariant convolutional neural networks.

![](https://user-images.githubusercontent.com/333780/73550102-425bd180-4444-11ea-8b69-8a4241ffa9c9.gif)

## Example
```python
from functools import partial
import torch
from e3nn.radial import CosineBasisModel
from e3nn.kernel import Kernel
from e3nn.point.operations import Convolution
from e3nn.util.plot import plot_sh_signal
import matplotlib.pyplot as plt

# Radial model:  R -> R^d
# Projection on cos^2 basis functions followed by a fully connected network
RadialModel = partial(CosineBasisModel, max_radius=3.0, number_of_basis=3, h=100, L=1, act=torch.relu)

# kernel: composed on a radial part that contains the learned parameters
#  and an angular part given by the spherical hamonics and the Clebsch-Gordan coefficients
K = partial(Kernel, RadialModel=RadialModel)

# Define input and output representations
Rs_in = [(1, 0)]  # one scalar
Rs_out = [(1, l) for l in range(10)]

# Use the kernel to define a convolution operation
conv = Convolution(K, Rs_in, Rs_out)

n = 3  # number of points
features = torch.ones(1, n, 1)
geometry = torch.randn(1, n, 3)

features = conv(features, geometry)
```

## Hierarchy

- `e3nn` contains the library
  - `e3nn/o3.py` defines all the needed mathematical functions
  - `e3nn/image` contains voxels linear operations
  - `e3nn/point` contains points linear operations
  - `e3nn/non_linearities` non linearities operations
- `examples` simple scripts and experiments

## Installation

1. install [pytorch](https://pytorch.org)
2. `pip install git+https://github.com/AMLab-Amsterdam/lie_learn`
3. `pip install git+https://github.com/e3nn/e3nn`

## Citing
[![DOI](https://zenodo.org/badge/116704656.svg)](https://zenodo.org/badge/latestdoi/116704656)

```
@software{mario_geiger_2020_3723557,
  author       = {Mario Geiger and
                  Tess Smidt and
                  Benjamin K. Miller and
                  Wouter Boomsma and
                  Kostiantyn Lapchevskyi and
                  Maurice Weiler and
                  Michał Tyszkiewicz and
                  Jes Frellsen},
  title        = {github.com/e3nn/e3nn},
  month        = mar,
  year         = 2020,
  publisher    = {Zenodo},
  version      = {v0.3-alpha},
  doi          = {10.5281/zenodo.3723557},
  url          = {https://doi.org/10.5281/zenodo.3723557}
}
```
