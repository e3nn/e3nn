from typing import Tuple

import torch
from torch import fx
import torch_scatter

from e3nn import o3
from e3nn.util.codegen import CodeGenMixin
from e3nn.util.jit import compile_mode
from ._tensor_product._codegen import _get_code


@compile_mode('script')
class Norm(CodeGenMixin, torch.nn.Module):
    r"""Norm of each irrep in a direct sum of irreps.

    Parameters
    ----------
    irreps_in : `Irreps`
        representation of the input

    Examples
    --------
    Compute the norms of 17 vectors.

    >>> norm = Norm("17x1o")
    >>> norm(torch.randn(17 * 3)).shape
    torch.Size([17])
    """
    def __init__(self, irreps_in):
        super().__init__()
        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_out = o3.Irreps([(mul, "0e") for mul, _ in self.irreps_in]).simplify()

        code, indptr = _codegen_norm(self.irreps_in)
        self._codegen_register({
            "_compiled_main": code
        })
        self.register_buffer("_indptr", indptr)

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
        return self._compiled_main(features, self._indptr)


def _codegen_norm(irreps_in: o3.Irreps,) -> Tuple[str, torch.Tensor]:
    graph_out = fx.Graph()

    # = Function definitions =
    x = fx.Proxy(graph_out.placeholder('x', torch.Tensor))
    indptr_proxy = fx.Proxy(graph_out.placeholder('indptr', torch.Tensor))

    size = x.shape[:-1]
    outsize = size + (irreps_in.num_irreps,)

    # = Short-circut for zero dimensional =
    if irreps_in.dim == 0:
        out = x.new_zeros(outsize)
        graph_out.output(out.node, torch.Tensor)
        # Short circut
        return _get_code(graph_out), torch.LongTensor([])

    x = x.reshape(-1, irreps_in.dim)

    # == Square ==
    x = torch.square(x)

    # = Scatter sums =
    indptr = [0]
    i = 0
    for mul_ir in irreps_in:
        for _ in range(mul_ir.mul):
            i += mul_ir.ir.dim
            indptr.append(i)
    assert len(indptr) == irreps_in.num_irreps + 1
    indptr = torch.LongTensor(indptr).view(1, -1)

    # fx can't trace this directly
    # https://pytorch-scatter.readthedocs.io/en/latest/functions/segment_csr.html
    # segment_csr is the most efficient way to do these sums
    # We call the ops version directly to (1) fix the FX function name issues and (2) avoid unnecessary nesting
    out = graph_out.call_function(
        torch.ops.torch_scatter.segment_sum_csr,
        args=(
            x.node,
            indptr_proxy.node,
            None
        )
    )
    out = fx.Proxy(out)
    out = torch.sqrt(out)

    # = Return the result =
    out = out.reshape(outsize)
    graph_out.output(out.node, torch.Tensor)

    # check graphs
    graph_out.lint()

    return _get_code(graph_out), indptr
