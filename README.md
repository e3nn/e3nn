# e3nn
[![Coverage Status](https://coveralls.io/repos/github/e3nn/e3nn/badge.svg?branch=master)](https://coveralls.io/github/e3nn/e3nn?branch=master)

E(3) is the [Euclidean group](https://en.wikipedia.org/wiki/Euclidean_group) in dimension 3. That is the group of rotations, translations and mirror.
`e3nn` is a [pytorch](https://pytorch.org) library that aims to create **E**(**3**) equivariant **n**eural **n**etworks.

![](https://user-images.githubusercontent.com/333780/79220728-dbe82c00-7e54-11ea-82c7-b3acbd9b2246.gif)

## Installation

After having installed [pytorch_geometric](https://github.com/rusty1s/pytorch_geometric#installation) run the command:

`pip install e3nn`

To get the CUDA kernels read the instructions in `INSTALL.md`.

## Example
```python
from functools import partial

import torch

from e3nn import Kernel, rs
from e3nn.non_linearities.norm import Norm
from e3nn.non_linearities.rescaled_act import swish
from e3nn.point.operations import Convolution
from e3nn.radial import GaussianRadialModel

# Define the input and output representations
Rs_in = [(1, 0), (2, 1)]  # Input = One scalar plus two vectors
Rs_out = [(1, 1)]  # Output = One single vector

# Radial model:  R+ -> R^d
RadialModel = partial(GaussianRadialModel, max_radius=3.0, number_of_basis=3, h=100, L=1, act=swish)

# kernel: composed on a radial part that contains the learned parameters
#  and an angular part given by the spherical hamonics and the Clebsch-Gordan coefficients
K = partial(Kernel, RadialModel=RadialModel)

# Create the convolution module
conv = Convolution(K(Rs_in, Rs_out))

# Module to compute the norm of each irreducible component
norm = Norm(Rs_out)


n = 5  # number of input points
features = rs.randn(1, n, Rs_in, requires_grad=True)
in_geometry = torch.randn(1, n, 3)
out_geometry = torch.zeros(1, 1, 3)  # One point at the origin


out = norm(conv(features, in_geometry, out_geometry))
out.backward()

print(out)
print(features.grad)
```

## Hierarchy

- `e3nn` contains the library
  - `e3nn/o3.py` O(3) irreducible representations
  - `e3nn/rsh.py` real spherical harmonics
  - `e3nn/rs.py` geometrical tensor representations
  - `e3nn/image` contains voxels linear operations
  - `e3nn/point` contains points linear operations
  - `e3nn/non_linearities` non linearities operations
- `examples` simple scripts and experiments

## Citing
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3723557.svg)](https://doi.org/10.5281/zenodo.3723557)

```
@software{e3nn_2020_3723557,
  author       = {Mario Geiger and
                  Tess Smidt and
                  Benjamin K. Miller and
                  Wouter Boomsma and
                  Kostiantyn Lapchevskyi and
                  Maurice Weiler and
                  Michał Tyszkiewicz and
                  Jes Frellsen},
  title        = {github.com/e3nn/e3nn},
  month        = may,
  year         = 2020,
  publisher    = {Zenodo},
  version      = {0.0.0},
  doi          = {10.5281/zenodo.3723557},
  url          = {https://doi.org/10.5281/zenodo.3723557}
}
```

### Copyright

Euclidean neural networks (e3nn) Copyright (c) 2020, The Regents of the
University of California, through Lawrence Berkeley National Laboratory
(subject to receipt of any required approvals from the U.S. Dept. of Energy), 
Ecole Polytechnique Federale de Lausanne (EPFL), Free University of Berlin 
and Kostiantyn Lapchevskyi. All rights reserved.

If you have questions about your rights to use or distribute this software,
please contact Berkeley Lab's Intellectual Property Office at
IPO@lbl.gov.

NOTICE.  This Software was developed under funding from the U.S. Department
of Energy and the U.S. Government consequently retains certain rights.  As
such, the U.S. Government has been granted for itself and others acting on
its behalf a paid-up, nonexclusive, irrevocable, worldwide license in the
Software to reproduce, distribute copies to the public, prepare derivative 
works, and perform publicly and display publicly, and to permit others to do so.
