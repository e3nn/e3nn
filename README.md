# E3NN
The group E(3) is the group of 3 dimensional rotations, translations and mirror.
This library aims to create [E(3) equivariant](https://youtu.be/ENLJACPHSEA) convolutional neural networks.

![](https://user-images.githubusercontent.com/333780/63736480-135b8700-c838-11e9-873f-2d65c67b98df.gif)

## Image and Point
The code is separated in two parts:
- `image` for volumetric data [[1]](https://arxiv.org/abs/1807.02547)
- `point` for point cloud data [[2]](https://arxiv.org/abs/1802.08219)

## Example
### Image
```python
import torch
from e3nn.image.convolution import SE3Convolution

size = 32  # space size

scalar_field = torch.randn(1, 1, size, size, size)  # [batch, _, x, y, z]

Rs_in = [(1, 0)]  # 1 scalar field
Rs_out = [(1, 1)]  # 1 vector field
conv = SE3Convolution(Rs_in, Rs_out, size=5)
# conv.weight.size() == [2] (2 radial degrees of freedom)

vector_field = conv(scalar_field)  # [batch, vector component, x, y, z]

# vector_field.size() == [1, 3, 28, 28, 28]
```
### Point
```python
from functools import partial
import torch
from e3nn.point.radial import CosineBasisModel
from e3nn.point.kernel import Kernel
from e3nn.point.operations import Convolution
from e3nn.util.plot import plot_sh_signal
import matplotlib.pyplot as plt

# Radial model:  R -> R^d
# Projection on cos^2 basis functions followed by a fully connected network
RadialModel = partial(CosineBasisModel, max_radius=3.0, number_of_basis=3, h=100, L=1, act=torch.relu)

# kernel: composed on a radial part that contains the learned parameters
#  and an angular part given by the spherical hamonics and the Clebsch-Gordan coefficients
K = partial(Kernel, RadialModel=RadialModel)

# Use the kernel to define a convolution operation
C = partial(Convolution, K)


Rs_in = [(1, 0)]  # one scalar
Rs_out = [(1, l) for l in range(10)]
conv = C(Rs_in, Rs_out)

n = 3  # number of points
features = torch.ones(1, n, 1)
geometry = torch.randn(1, n, 3)

features = conv(features, geometry)

plt.figure(figsize=(4, 4))
plot_sh_signal(features[:, 0], n=50)
plt.gca().view_init(azim=0, elev=45)
```
![](https://user-images.githubusercontent.com/333780/61739910-43e46880-ad42-11e9-97d6-ecbe71affd2e.png)

## Hierarchy

- `e3nn` contains the library
  - `e3nn/SO3.py` defines all the needed mathematical functions
  - `e3nn/image` contains the code specific to voxels
  - `e3nn/point` contains the code specific to points
  - `e3nn/non_linearities` non linearities working for both point and voxel code
- `examples` simple scripts and experiments

## Installation

1. install [pytorch](https://pytorch.org)
2. `pip install git+https://github.com/AMLab-Amsterdam/lie_learn`
3. `pip install git+https://github.com/e3nn/e3nn`

## Usage

Install with
```
python setup.py install
```

## Citing
[![DOI](https://zenodo.org/badge/116704656.svg)](https://zenodo.org/badge/latestdoi/116704656)

```
@misc{mario_geiger_2019_3348277,
  author       = {Mario Geiger and
                  Tess Smidt and
                  Wouter Boomsma and
                  Maurice Weiler and
                  Michał Tyszkiewicz and
                  Jes Frellsen and
                  Benjamin K. Miller},
  title        = {mariogeiger/e3nn: Point cloud support},
  month        = jul,
  year         = 2019,
  doi          = {10.5281/zenodo.3348277},
  url          = {https://doi.org/10.5281/zenodo.3348277}
}
```
