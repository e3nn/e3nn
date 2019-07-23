# SE3CNN
The group SE(3) is the group of 3 dimensional rotations and translations.
This library aims to create [SE(3) equivariant](https://youtu.be/ENLJACPHSEA) convolutional neural networks.

![](https://github.com/antigol/se3net/raw/master/examples/plots/kernels.png)

## Image and Point
The code is separated in two parts:
- `image` for volumetric data [[1]](https://arxiv.org/abs/1807.02547) 
- `point` for point cloud data [[2]](https://arxiv.org/abs/1802.08219)

## Example
### Image
```python
import torch
from se3cnn.image.convolution import SE3Convolution

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
from se3cnn.point.radial import CosineBasisModel
from se3cnn.point.kernel import SE3PointKernel
from se3cnn.point.convolution import SE3PointConvolution
from se3cnn.util.plot import plot_sh_signal
import matplotlib.pyplot as plt

# Radial model:  R -> R^d
# Projection on cos^2 basis functions followed by a fully connected network
RadialModel = partial(CosineBasisModel, max_radius=3.0, number_of_basis=3, h=100, L=1, act=torch.relu)

# kernel: composed on a radial part that contains the learned parameters 
#  and an angular part given by the spherical hamonics and the Clebsch-Gordan coefficients
Kernel = partial(SE3PointKernel, RadialModel=RadialModel)

# Use the kernel to define a convolution operation
Convolution = partial(SE3PointConvolution, Kernel)


Rs_in = [(1, 0)]  # one scalar
Rs_out = [(1, l) for l in range(10)]
conv = Convolution(Rs_in, Rs_out)

n = 3  # number of points
features = torch.ones(1, n)
geometry = torch.randn(n, 3)

features = conv(features, geometry)

plt.figure(figsize=(4, 4))
plot_sh_signal(features[:, 0], n=50)
plt.gca().view_init(azim=0, elev=45)
```
![](https://user-images.githubusercontent.com/333780/61739910-43e46880-ad42-11e9-97d6-ecbe71affd2e.png)

## Hierarchy

- `se3cnn` contains the library
  - `se3cnn/SO3.py` defines all the needed mathematical functions
  - `se3cnn/image` contains the code specific to voxels
  - `se3cnn/point` contains the code specific to points
  - `se3cnn/non_linearities` non linearities working for both point and voxel code
- `examples` simple scripts and experiments

## Dependencies

- [pytorch](https://pytorch.org) (v >= 1)
- [lie_learn](https://github.com/AMLab-Amsterdam/lie_learn) is required to compute the irreducible representations of SO(3)
- scipy

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
                  Ben},
  title        = {mariogeiger/se3cnn: Point cloud support},
  month        = jul,
  year         = 2019,
  doi          = {10.5281/zenodo.3348277},
  url          = {https://doi.org/10.5281/zenodo.3348277}
}
```
