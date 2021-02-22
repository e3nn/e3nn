import torch

from e3nn import o3


class Linear(torch.nn.Module):
    r"""Linear operation equivariant to :math:`O(3)`

    Parameters
    ----------
    irreps_in : `Irreps`
        representation of the input

    irreps_out : `Irreps`
        representation of the output

    internal_weights : bool
        see `TensorProduct`

    shared_weights : bool
        see `TensorProduct`

    Examples
    --------
    Linearly combines 4 scalars into 8 scalars and 16 vectors into 8 vectors.

    >>> lin = Linear("4x0e+16x1o", "8x0e+8x1o")
    >>> lin.tp.weight_numel
    160
    """
    def __init__(
            self,
            irreps_in,
            irreps_out,
            internal_weights=None,
            shared_weights=None,
                ):
        super().__init__()

        irreps_in = o3.Irreps(irreps_in).simplify()
        irreps_out = o3.Irreps(irreps_out).simplify()

        instr = [
            (i_in, 0, i_out, 'uvw', True, 1.0)
            for i_in, (_, ir_in) in enumerate(irreps_in)
            for i_out, (_, ir_out) in enumerate(irreps_out)
            if ir_in == ir_out
        ]

        self.tp = o3.TensorProduct(irreps_in, "0e", irreps_out, instr, internal_weights=internal_weights, shared_weights=shared_weights)

        self.output_mask = self.tp.output_mask
        self.irreps_in = irreps_in
        self.irreps_out = irreps_out

    def __repr__(self):
        return f"{self.__class__.__name__}({self.irreps_in} -> {self.irreps_out} | {self.tp.weight_numel} weights)"

    def forward(self, features, weight=None):
        """evaluate

        Parameters
        ----------
        features : `torch.Tensor`
            tensor of shape ``(..., irreps_in.dim)``

        weight : `torch.Tensor`, optional
            required if ``internal_weights`` is `False`

        Returns
        -------
        `torch.Tensor`
            tensor of shape ``(..., irreps_out.dim)``
        """
        ones = torch.ones(features.shape[:-1] + (1,), dtype=features.dtype, device=features.device)
        return self.tp(features, ones, weight)
