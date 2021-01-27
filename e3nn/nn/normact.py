import torch

from e3nn.o3 import ElementwiseTensorProduct, Norm

# TODO: test this!
class NormActivation(torch.nn.Module):
    r"""Gate activation function

    Parameters
    ----------
    irreps_scalars : `Irreps`
        representation of the scalars (the one not use for the gates)

    act_scalars : list of function or None
        activations acting on the scalars

    irreps_gates : `Irreps`
        representation of the scalars used to gate.
        ``irreps_gates.num_irreps == irreps_nonscalars.num_irreps``

    act_gates : list of function or None
        activations acting on the gates

    irreps_nonscalars : `Irreps`
        representation of the non-scalars.
        ``irreps_gates.num_irreps == irreps_nonscalars.num_irreps``

    Examples
    --------

    >>> g = Gate("16x0o", [torch.tanh], "32x0o", [torch.tanh], "16x1e+16x1o")
    >>> g.irreps_out
    16x0o+16x1o+16x1e
    """
    def __init__(self,
                 irreps_in,
                 scalar_nonlinearity,
                 normalize=True,
                 bias=False,
                 epsilon=1e-8,
                 **kwargs):
        super().__init__()
        self.irreps_in = irreps_in
        self.irreps_out = irreps_in
        self.norm = Norm(irreps_in, **kwargs)
        self.scalar_nonlinearity = scalar_nonlinearity
        self.epsilon = torch.as_tensor(epsilon)
        self.normalize = True
        self.bias = bias
        if self.bias:
            self.biases = torch.nn.Parameter(torch.zeros(irreps_in.num_irreps))

        self.scalar_multiplier = ElementwiseTensorProduct(
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
            tensor of shape ``(..., irreps_out.dim)``
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
            scalings = self.scalar_nonlinearity(norms)

        return self.scalar_multiplier(scalings, features)
