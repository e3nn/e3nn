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

    epsilon : float
        when ``normalize`ing, norms smaller than ``epsilon`` will be clamped to ``epsilon`` to avoid division by zero

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
    def __init__(self,
                 irreps_in,
                 scalar_nonlinearity,
                 normalize=True,
                 epsilon=1e-8,
                 bias=False):
        super().__init__()
        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_out = o3.Irreps(irreps_in)
        self.norm = o3.Norm(irreps_in)
        self.scalar_nonlinearity = scalar_nonlinearity
        self.register_buffer('epsilon', torch.as_tensor(epsilon))
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

        # Apply nonlinearity
        if self.normalize:
            epsilon_norms = norms.clone()
            epsilon_norms[epsilon_norms < self.epsilon] = self.epsilon
            nonlin_arg = epsilon_norms
            if self.bias:
                nonlin_arg = nonlin_arg + self.biases
            scalings = self.scalar_nonlinearity(nonlin_arg) / epsilon_norms
        else:
            nonlin_arg = norms
            if self.bias:
                nonlin_arg = nonlin_arg + self.biases
            scalings = self.scalar_nonlinearity(nonlin_arg)

        return self.scalar_multiplier(scalings, features)
