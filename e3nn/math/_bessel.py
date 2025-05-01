# Porting from https://github.com/e3nn/e3nn-jax/blob/403f0ef159c537c2efa9d3d05799da85a86575d5/e3nn_jax/_src/radial.py#L213-L244

import torch
import numpy as np

def bessel(x: torch.Tensor,
           n: int,
           x_max: float=1.0) -> torch.Tensor:
    r"""Bessel basis functions.

        They obey the following normalization:

        .. math::

            \int_0^c r^2 B_n(r, c) B_m(r, c) dr = \delta_{nm}

        Args:
            x (torch.Tensor): input of shape ``[...]``
            n (int): number of basis functions
            x_max (float): maximum value of the input

        Returns:
            torch.Tensor: basis functions of shape ``[..., n]``

        Klicpera, J.; Groß, J.; Günnemann, S. Directional Message Passing for Molecular Graphs; ICLR 2020.
        Equation (7)
        """
    assert isinstance(n, int)
    
    x = x[..., None]
    n = torch.arange(1, n + 1, dtype=x.dtype, device=x.device)
    x_nonzero = torch.where(x == 0.0, 1.0, x)
    return np.sqrt(2.0 / x_max) * torch.where(
        x == 0,
        n * torch.pi / x_max,
        torch.sin(n * torch.pi / x_max * x_nonzero) / x_nonzero
    )