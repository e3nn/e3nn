from typing import Optional, Tuple, NamedTuple, List
import math

import torch
from torch import fx

import e3nn
from e3nn import o3
from e3nn.util.jit import compile_mode
from e3nn.util.codegen import CodeGenMixin
from e3nn.util import prod
from ._tensor_product._codegen import _sum_tensors


class Instruction(NamedTuple):
    i_in: int
    i_out: int
    path_shape: tuple


@compile_mode('script')
class Linear(CodeGenMixin, torch.nn.Module):
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

    instructions : list of 2-tuples, optional
        list of tuples ``(i_in, i_out)`` indicating which irreps in ``irreps_in`` should contribute to which irreps in ``irreps_out``. If ``None`` (the default), all allowable instructions will be created: every ``(i_in, i_out)`` such that ``irreps_in[i_in].ir == irreps_out[i_out].ir``.

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

    Create a "block sparse" linear that does not combine two different groups of scalars;
    note that the number of weights is 4*4 + 3*3 = 25:

    >>> lin = Linear("4x0e + 3x0e", "4x0e + 3x0e", instructions=[(0, 0), (1, 1)])
    >>> lin.weight_numel
    25

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
        instructions: Optional[List[Tuple[int, int]]] = None,
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

        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_out = o3.Irreps(irreps_out)
        del irreps_in
        del irreps_out

        opt_defaults = e3nn.get_optimization_defaults()
        self._optimize_einsums = _optimize_einsums if _optimize_einsums is not None else opt_defaults['optimize_einsums']
        del opt_defaults

        # == Instructions ==
        if instructions is None:
            # In the fully connected case, everything goes everywhere anyway,
            # so we can simplify the irreps and maybe save some compute:
            self.irreps_in = self.irreps_in.simplify()
            self.irreps_out = self.irreps_out.simplify()
            # ^ we can't do this in the general case; see the block sparse example.

            # By default, make all possible connections
            instructions = [
                (i_in, i_out)
                for i_in, (_, ir_in) in enumerate(self.irreps_in)
                for i_out, (_, ir_out) in enumerate(self.irreps_out)
                if ir_in == ir_out
            ]

        instruction_objs = []
        for i_in, i_out in instructions:
            assert isinstance(i_in, int) and isinstance(i_out, int)
            if self.irreps_in[i_in].ir != self.irreps_out[i_out].ir:
                raise ValueError(
                    f"Invalid instruction to connect irreps_in[{i_in}] = {self.irreps_in[i_in]} to different irrep irreps_out[{i_out}] = {self.irreps_out[i_out]}"
                )
            instruction_objs.append(
                Instruction(
                    i_in=i_in,
                    i_out=i_out,
                    path_shape=(self.irreps_in[i_in].mul, self.irreps_out[i_out].mul)
                )
            )
        self.instructions = instruction_objs
        del instructions
        del instruction_objs

        # == Generate code ==
        graph, self.weight_numel = _codegen_linear(
            self.irreps_in,
            self.irreps_out,
            self.instructions,
            shared_weights=shared_weights,
            optimize_einsums=self._optimize_einsums
        )
        self._codegen_register({
            "_compiled_main": graph
        })

        # == Generate weights ==
        if internal_weights and self.weight_numel > 0:
            assert self.shared_weights, "Having internal weights impose shared weights"
            self.weight = torch.nn.Parameter(torch.randn(self.weight_numel))
        else:
            # For TorchScript, there always has to be some kind of defined .weight
            self.register_buffer('weight', torch.Tensor())

        # == Compute output mask ==
        if self.irreps_out.dim > 0:
            output_mask = torch.cat([
                torch.ones(mul_ir.dim)
                if any(ins.i_out == i_out for ins in self.instructions)
                else torch.zeros(mul_ir.dim)
                for i_out, mul_ir in enumerate(self.irreps_out)
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
    instructions: List[Instruction],
    shared_weights: bool = False,
    optimize_einsums: bool = True,
) -> Tuple[fx.Graph, int]:
    graph_out = fx.Graph()

    # = Function definitions =
    x = fx.Proxy(graph_out.placeholder('x', torch.Tensor))
    ws = fx.Proxy(graph_out.placeholder('w', torch.Tensor))

    size = x.shape[:-1]
    outsize = size + (irreps_out.dim,)

    # = Short-circut for nothing to do =
    if len(instructions) == 0:
        out = x.new_zeros(outsize)

        graph_out.output(out.node, torch.Tensor)
        # Short circut
        # 0 is weight_numel
        return graph_out, 0

    x = x.reshape(-1, irreps_in.dim)
    batch_out = x.shape[0]

    weight_numel = sum(prod(ins.path_shape) for ins in instructions)
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

    # TODO: ?? "fuse" instructions to make more efficient? I.e.:
    # instead of (0, 0), (0, 1), etc., do one big einsum of input 0 with many weights,
    # and then extract the relevant parts for summing the outputs.

    for ins in instructions:
        mul_ir_in = irreps_in[ins.i_in]
        mul_ir_out = irreps_out[ins.i_out]

        # Short-circut for empty irreps
        if mul_ir_in.dim == 0 or mul_ir_out.dim == 0:
            continue

        # Extract the weight from the flattened weight tensor
        path_nweight = prod(ins.path_shape)
        if len(instructions) == 1:
            # Avoid unnecessary view when there is only one weight
            w = ws
        else:
            w = ws[
                :,
                flat_weight_index:flat_weight_index + path_nweight
            ]
        w = w.reshape(
            (() if shared_weights else (-1,)) + ins.path_shape
        )
        flat_weight_index += path_nweight

        ein_out = torch.einsum(f"{z}uw,zui->zwi", w, x_list[ins.i_in])
        alpha = 1.0 / math.sqrt(
            mul_ir_in.mul * sum(
                1 if other_ins.i_out == ins.i_out else 0
                for other_ins in instructions
            )
        )
        ein_out = alpha * ein_out

        out_list += [ein_out.reshape(batch_out, mul_ir_out.dim)]

    # = Return the result =
    out = [
        _sum_tensors(
            [out for ins, out in zip(instructions, out_list) if ins.i_out == i_out],
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
