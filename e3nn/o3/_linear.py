from typing import Optional, Tuple
import math

import torch
from torch import fx

import e3nn
from e3nn import o3
from e3nn.util.jit import compile_mode
from ._tensor_product._codegen import _sum_tensors


@compile_mode('script')
class Linear(torch.nn.Module):
    r"""Linear operation equivariant to :math:`O(3)`

    Parameters
    ----------
    irreps_in : `Irreps`
        representation of the input

    irreps_out : `Irreps`
        representation of the output

    internal_weights : bool
        whether the ``Linear`` should store its own weights. Defaults to ``True`` unless ``shared_weights`` is explicitly set to ``False``, for consistancy with ``TensorProduct``.

    shared_weights : bool
        whether the ``Linear`` should be weighted individually for each input in a batch. Defaults to ``False``. Cannot be ``True`` if ``internal_weights`` is ``True``.

    Attributes
    ----------
    weight_numel : int
        the size of the weights for this ``Linear``

    Examples
    --------
    Linearly combines 4 scalars into 8 scalars and 16 vectors into 8 vectors.

    >>> lin = Linear("4x0e+16x1o", "8x0e+8x1o")
    >>> lin.weight_numel
    160
    """
    weight_numel: int
    internal_weights: bool
    shared_weights: bool

    def __init__(
        self,
        irreps_in,
        irreps_out,
        internal_weights: Optional[bool] = None,
        shared_weights: Optional[bool] = None,
        _optimize_einsums: Optional[bool] = None
    ):
        super().__init__()

        # == Process arguments ==
        if shared_weights is False and internal_weights is None:
            internal_weights = False

        if shared_weights is None:
            shared_weights = True

        if internal_weights is None:
            internal_weights = True

        assert shared_weights or not internal_weights
        self.internal_weights = internal_weights
        self.shared_weights = shared_weights

        self.irreps_in = o3.Irreps(irreps_in).simplify()
        self.irreps_out = o3.Irreps(irreps_out).simplify()

        opt_defaults = e3nn.get_optimization_defaults()
        self._optimize_einsums = _optimize_einsums if _optimize_einsums is not None else opt_defaults['optimize_einsums']
        del opt_defaults

        # == Generate code ==
        graph, self.weight_numel = _codegen_linear(
            self.irreps_in,
            self.irreps_out,
            shared_weights=shared_weights,
            optimize_einsums=self._optimize_einsums
        )
        self._compiled_main = torch.jit.script(fx.GraphModule(
            root=self,
            graph=graph
        ))

        # == Generate weights ==
        if internal_weights and self.weight_numel > 0:
            assert self.shared_weights, "Having internal weights impose shared weights"
            self.weight = torch.nn.Parameter(torch.randn(self.weight_numel))
        else:
            # For TorchScript, there always has to be some kind of defined .weight
            self.register_buffer('weight', torch.Tensor())

        # == Compute output mask ==
        if self.irreps_out.dim > 0:
            # an output is connected to an input if one exists with the same L, p
            output_mask = torch.cat([
                torch.ones(mul * ir.dim)
                if any(ir_in == ir for _, ir_in in self.irreps_in)
                else torch.zeros(mul * ir.dim)
                for _, (mul, ir) in enumerate(self.irreps_out)
            ])
        else:
            output_mask = torch.ones(0)
        self.register_buffer('output_mask', output_mask)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.irreps_in} -> {self.irreps_out} | {self.weight_numel} weights)"

    def forward(self, features, weight: Optional[torch.Tensor] = None):
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
        if weight is None:
            assert self.internal_weights, "Weights must be provided when internal_weights = False"
            weight = self.weight
        return self._compiled_main(features, weight)


def _codegen_linear(
    irreps_in: o3.Irreps,
    irreps_out: o3.Irreps,
    shared_weights: bool = False,
    optimize_einsums: bool = True,
) -> Tuple[fx.Graph, int]:
    graph_out = fx.Graph()

    # = Function definitions =
    x = fx.Proxy(graph_out.placeholder('x', torch.Tensor))
    ws = fx.Proxy(graph_out.placeholder('w', torch.Tensor))

    size = x.shape[:-1]
    outsize = size + (irreps_out.dim,)

    # = Short-circut for zero dimensional =
    if irreps_in.dim == 0 or irreps_out.dim == 0:
        out = x.new_zeros(outsize)

        graph_out.output(out.node, torch.Tensor)
        # Short circut
        # 0 is weight_numel
        return graph_out, 0

    x = x.reshape(-1, irreps_in.dim)
    batch_out = x.shape[0]

    weight_numel = sum(
        mul_in * mul_out
        for mul_in, ir_in in irreps_in
        for mul_out, ir_out in irreps_out
        if ir_in == ir_out
    )
    if weight_numel > 0:
        ws = ws.reshape(-1, weight_numel)

    # = extract individual input irreps =
    if len(irreps_in) == 1:
        x_list = [x.reshape(batch_out, irreps_in[0].mul, irreps_in[0].ir.dim)]
    else:
        x_list = [
            x[:, i].reshape(batch_out, mul_ir.mul, mul_ir.ir.dim)
            for i, mul_ir in zip(irreps_in.slices(), irreps_in)
        ]

    z = '' if shared_weights else 'z'

    flat_weight_index = 0

    out_list = []

    instr = [
        (i_in, i_out)
        for i_in, (_, ir_in) in enumerate(irreps_in)
        for i_out, (_, ir_out) in enumerate(irreps_out)
        if ir_in == ir_out
    ]

    for i_in, i_out in instr:
        mul_ir_in = irreps_in[i_in]
        mul_ir_out = irreps_out[i_out]

        # Short-circut for empty irreps
        if mul_ir_in.dim == 0 or mul_ir_out.dim == 0:
            continue

        # Extract the weight from the flattened weight tensor
        path_nweight = mul_ir_in.mul*mul_ir_out.mul
        if len(instr) == 1:
            w = ws
        else:
            w = ws[
                :,
                flat_weight_index:flat_weight_index + path_nweight
            ]
        w = w.reshape(
            (() if shared_weights else (-1,)) + (mul_ir_in.mul, mul_ir_out.mul)
        )
        flat_weight_index += path_nweight

        ein_out = torch.einsum(f"{z}uw,zui->zwi", w, x_list[i_in])
        alpha = 1.0 / math.sqrt(mul_ir_in.mul * sum(1 if i_out_this == i_out else 0 for _, i_out_this in instr))
        ein_out = alpha * ein_out

        out_list += [ein_out.reshape(batch_out, mul_ir_out.dim)]

    # = Return the result =
    out = [
        _sum_tensors(
            [out for (_, i_out_ins), out in zip(instr, out_list) if i_out_ins == i_out],
            shape=(batch_out, mul_ir_out.dim),
            like=x
        )
        for i_out, mul_ir_out in enumerate(irreps_out)
    ]
    if len(out) > 1:
        out = torch.cat(out, dim=1)
    else:
        out = out[0]

    out = out.reshape(outsize)

    graph_out.output(out.node, torch.Tensor)

    # check graphs
    graph_out.lint()

    # TODO: when eliminate_dead_code() is in PyTorch stable, use that
    if optimize_einsums:
        try:
            from opt_einsum_fx import optimize_einsums_full, jitable
        except ImportError:
            # opt_einsum_fx is not installed
            pass
        else:
            # See _tensor_product/_codegen.py for notes
            batchdim = 4
            example_inputs = (
                torch.zeros((batchdim, irreps_in.dim), dtype=torch.float32),
                torch.zeros(
                    1 if shared_weights else batchdim,
                    flat_weight_index,
                    dtype=torch.float32
                ),
            )
            graph_out = jitable(optimize_einsums_full(graph_out, example_inputs))

    return graph_out, weight_numel
