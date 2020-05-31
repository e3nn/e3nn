
# pylint: disable=missing-docstring, line-too-long, invalid-name, arguments-differ, no-member, pointless-statement
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
