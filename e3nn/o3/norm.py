import torch

from e3nn import o3


class Norm(torch.nn.Module):
    r"""Norm operation

    Parameters
    ----------
    irreps_in : `Irreps`
        representation of the input

    normalization : {'component', 'norm'}
        see `TensorProduct`

    Examples
    --------
    Compute the norms of 17 vectors.

    >>> norm = Norm("17x1o")
    >>> norm(torch.randn(17 * 3)).shape
    torch.Size([17])
    """
    def __init__(
            self,
            irreps_in,
                ):
        super().__init__()

        irreps_in = o3.Irreps(irreps_in).simplify()
        irreps_out = o3.Irreps([(mul, "0e") for mul, _ in irreps_in])

        instr = [
            (i, i, i, 'uuu', False, ir.dim)
            for i, (mul, ir) in enumerate(irreps_in)
        ]

        self.tp = o3.TensorProduct(irreps_in, irreps_in, irreps_out, instr, 'component')

        self.irreps_in = irreps_in
        self.irreps_out = irreps_out.simplify()

    def __repr__(self):
        return f"{self.__class__.__name__}({self.irreps_in})"

    def forward(self, features):
        """evaluate

        Parameters
        ----------
        features : `torch.Tensor`
            tensor of shape ``(..., irreps_in.dim)``

        Returns
        -------
        `torch.Tensor`
            tensor of shape ``(..., irreps_out.dim)``
        """
        return self.tp(features, features).sqrt()
