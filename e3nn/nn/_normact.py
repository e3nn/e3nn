from typing import Callable, Optional

import torch

from e3nn import o3
from e3nn.util.jit import compile_mode


@compile_mode('trace')
class NormActivation(torch.nn.Module):
    r"""Norm-based activation function

    Applies a scalar nonlinearity to the norm of each irrep and ouputs a (normalized) version of that irrep multiplied by the scalar output of the scalar nonlinearity.


    Parameters
    ----------
    irreps_in : `Irreps`
        representation of the input

    scalar_nonlinearity : callable
        scalar nonlinearity such as ``torch.sigmoid``

    normalize : bool
        whether to normalize the input features before multiplying them by the scalars from the nonlinearity

    epsilon : float, optional
        when ``normalize``ing, norms smaller than ``epsilon`` will be clamped up to ``epsilon`` to avoid division by zero. See also ``epsilon`` parameter to ``o3.Norm``. Cannot be specified if ``normalize`` is ``False``.

    bias : bool
        whether to apply a learnable additive bias to the inputs of the ``scalar_nonlinearity``

    Examples
    --------

    >>> n = NormActivation("2x1e", torch.sigmoid)
    >>> feats = torch.ones(1, 2*3)
    >>> print(feats.reshape(1, 2, 3).norm(dim=-1))
    tensor([[1.7321, 1.7321]])
    >>> print(torch.sigmoid(feats.reshape(1, 2, 3).norm(dim=-1)))
    tensor([[0.8497, 0.8497]])
    >>> print(n(feats).reshape(1, 2, 3).norm(dim=-1))
    tensor([[0.8497, 0.8497]])
    """
    def __init__(
        self,
        irreps_in,
        scalar_nonlinearity: Callable,
        normalize: bool = True,
        epsilon: Optional[float] = None,
        bias: bool = False
    ):
        super().__init__()
        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_out = o3.Irreps(irreps_in)

        if normalize:
            if epsilon is None:
                epsilon = 1e-8
            elif not epsilon > 0:
                raise ValueError(f"epsilon {epsilon} is invalid, must be strictly positive.")
        elif epsilon is not None:
            raise ValueError("Using epsilon when normalize = False makes no sense.")

        self.norm = o3.Norm(irreps_in, epsilon=epsilon)
        self.scalar_nonlinearity = scalar_nonlinearity
        self.normalize = normalize
        self.bias = bias
        if self.bias:
            self.biases = torch.nn.Parameter(torch.zeros(irreps_in.num_irreps))

        self.scalar_multiplier = o3.ElementwiseTensorProduct(
            irreps_in1=self.norm.irreps_out,
            irreps_in2=irreps_in,
        )

    def forward(self, features):
        '''evaluate

        Parameters
        ----------
        features : `torch.Tensor`
            tensor of shape ``(..., irreps_in.dim)``

        Returns
        -------
        `torch.Tensor`
            tensor of shape ``(..., irreps_in.dim)``
        '''
        norms = self.norm(features)

        nonlin_arg = norms
        if self.bias:
            nonlin_arg = nonlin_arg + self.biases

        scalings = self.scalar_nonlinearity(nonlin_arg)
        if self.normalize:
            scalings = scalings / norms

        return self.scalar_multiplier(scalings, features)
