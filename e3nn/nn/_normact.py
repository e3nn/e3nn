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
        when ``normalize``ing, norms smaller than ``epsilon`` will be clamped up to ``epsilon`` to avoid division by zero and NaN gradients. Not allowed when ``normalize`` is False.
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
    epsilon: Optional[float]
    _eps_squared: float

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

        if epsilon is None and normalize:
            epsilon = 1e-8
        elif epsilon is not None and not normalize:
            raise ValueError("epsilon and normalize = False don't make sense together")
        elif not epsilon > 0:
            raise ValueError(f"epsilon {epsilon} is invalid, must be strictly positive.")
        self.epsilon = epsilon
        if self.epsilon is not None:
            self._eps_squared = epsilon * epsilon
        else:
            self._eps_squared = 0.0  # doesn't matter

        # if we have an epsilon, use squared and do the sqrt ourselves
        self.norm = o3.Norm(irreps_in, squared=(epsilon is not None))
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
        if self._eps_squared > 0:
            # See TFN for the original version of this approach:
            # https://github.com/tensorfieldnetworks/tensorfieldnetworks/blob/master/tensorfieldnetworks/utils.py#L22
            norms[norms < self._eps_squared] = self._eps_squared
            norms = norms.sqrt()

        nonlin_arg = norms
        if self.bias:
            nonlin_arg = nonlin_arg + self.biases

        scalings = self.scalar_nonlinearity(nonlin_arg)
        if self.normalize:
            scalings = scalings / norms

        return self.scalar_multiplier(scalings, features)
