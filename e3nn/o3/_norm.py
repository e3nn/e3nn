import torch

from e3nn import o3
from e3nn.util.jit import compile_mode


@compile_mode('trace')
class Norm(torch.nn.Module):
    r"""Norm of each irrep in a direct sum of irreps.

    Parameters
    ----------
    irreps_in : `Irreps`
        representation of the input

    squared : bool, optional
        Whether to return the squared norm. ``False`` by default, i.e. the norm itself (sqrt of squared norm) is returned.

    Examples
    --------
    Compute the norms of 17 vectors.

    >>> norm = Norm("17x1o")
    >>> norm(torch.randn(17 * 3)).shape
    torch.Size([17])
    """
    squared: bool

    def __init__(
        self,
        irreps_in,
        squared: bool = False
    ):
        super().__init__()

        irreps_in = o3.Irreps(irreps_in).simplify()
        irreps_out = o3.Irreps([(mul, "0e") for mul, _ in irreps_in])

        instr = [
            (i, i, i, 'uuu', False, ir.dim)
            for i, (mul, ir) in enumerate(irreps_in)
        ]

        self.tp = o3.TensorProduct(
            irreps_in,
            irreps_in,
            irreps_out,
            instr,
            normalization='component'
        )

        self.irreps_in = irreps_in
        self.irreps_out = irreps_out.simplify()
        self.squared = squared

    def __repr__(self):
        return f"{self.__class__.__name__}({self.irreps_in})"

    def forward(self, features):
        """Compute norms of irreps in ``features``.

        Parameters
        ----------
        features : `torch.Tensor`
            tensor of shape ``(..., irreps_in.dim)``

        Returns
        -------
        `torch.Tensor`
            tensor of shape ``(..., irreps_out.dim)``
        """
        out = self.tp(features, features)
        if self.squared:
            return out
        else:
            # ReLU fixes gradients at zero
            return out.relu().sqrt()
