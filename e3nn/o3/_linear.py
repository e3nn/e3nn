from typing import List, NamedTuple, Optional, Tuple, Union

from opt_einsum_fx import optimize_einsums_full
import torch
from torch import fx

import e3nn
from e3nn import o3
from e3nn.util import prod
from e3nn.util.codegen import CodeGenMixin
from e3nn.util.jit import compile_mode

from ._tensor_product._codegen import _sum_tensors


class Instruction(NamedTuple):
    i_in: int
    i_out: int
    path_shape: tuple
    path_weight: float


@compile_mode('script')
class Linear(CodeGenMixin, torch.nn.Module):
    r"""Linear operation equivariant to :math:`O(3)`

    Notes
    -----
        `e3nn.o3.Linear` objects created with different partitionings of the same irreps, such as ``Linear("10x0e", "0e")`` and ``Linear("3x0e + 7x0e", "0e")``, are *not* equivalent: the second module has more instructions, which affects normalization. In a rough sense:

            Linear("10x0e", "0e") = normalization_coeff_0 * W_0 @ input
            Linear("3x0e + 7x0e", "0e") = normalization_coeff_1 * W_1 @ input[:3] + normalization_coeff_2 * W_2 @ input[3:]

        To make them equivalent, simplify ``irreps_in`` before constructing network modules:

            o3.Irreps("3x0e + 7x0e").simplify()  # => 10x0e


    Parameters
    ----------
    irreps_in : `e3nn.o3.Irreps`
        representation of the input

    irreps_out : `e3nn.o3.Irreps`
        representation of the output

    internal_weights : bool
        whether the `e3nn.o3.Linear` should store its own weights. Defaults to ``True`` unless ``shared_weights`` is explicitly set to ``False``, for consistancy with `e3nn.o3.TensorProduct`.

    shared_weights : bool
        whether the `e3nn.o3.Linear` should be weighted individually for each input in a batch. Defaults to ``False``. Cannot be ``True`` if ``internal_weights`` is ``True``.

    instructions : list of 2-tuples, optional
        list of tuples ``(i_in, i_out)`` indicating which irreps in ``irreps_in`` should contribute to which irreps in ``irreps_out``. If ``None`` (the default), all allowable instructions will be created: every ``(i_in, i_out)`` such that ``irreps_in[i_in].ir == irreps_out[i_out].ir``.

    biases : list of bool, optional
        indicates for each element of ``irreps_out`` if it has a bias. By default there is no bias. If ``biases=True`` it gives bias to all scalars (l=0 and p=1).

    Attributes
    ----------
    weight_numel : int
        the size of the weights for this `e3nn.o3.Linear`

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

    Be careful: because they have different instructions, the following two operations are not normalized in the same way, even though they contain all the same "connections":

    >>> lin1 = Linear("10x0e", "0e")
    >>> lin2 = Linear("3x0e + 7x0e", "0e")
    >>> lin1.weight_numel == lin2.weight_numel
    True
    >>> with torch.no_grad():
    ...     lin1.weight.fill_(1.0)
    ...     lin2.weight.fill_(1.0)
    Parameter containing:
    ...
    >>> x = torch.arange(10.0)
    >>> (lin1(x) - lin2(x)).abs().item() < 1e-5
    True

    """
    weight_numel: int
    internal_weights: bool
    shared_weights: bool

    def __init__(
        self,
        irreps_in,
        irreps_out,
        *,
        f_in: Optional[int] = None,
        f_out: Optional[int] = None,
        internal_weights: Optional[bool] = None,
        shared_weights: Optional[bool] = None,
        instructions: Optional[List[Tuple[int, int]]] = None,
        biases: Union[bool, List[bool]] = False,
        path_normalization: str = 'element',
        _optimize_einsums: Optional[bool] = None
    ):
        super().__init__()

        assert path_normalization in ['element', 'path']

        irreps_in = o3.Irreps(irreps_in)
        irreps_out = o3.Irreps(irreps_out)

        if instructions is None:
            # By default, make all possible connections
            instructions = [
                (i_in, i_out)
                for i_in, (_, ir_in) in enumerate(irreps_in)
                for i_out, (_, ir_out) in enumerate(irreps_out)
                if ir_in == ir_out
            ]

        instructions = [
            Instruction(
                i_in=i_in,
                i_out=i_out,
                path_shape=(irreps_in[i_in].mul, irreps_out[i_out].mul),
                path_weight=1,
            )
            for i_in, i_out in instructions
        ]

        def alpha(ins):
            x = sum(
                irreps_in[i.i_in if path_normalization == 'element' else ins.i_in].mul
                for i in instructions
                if i.i_out == ins.i_out
            )
            if f_in is not None:
                x *= f_in
            return 1.0 if x == 0 else x

        instructions = [
            Instruction(
                i_in=ins.i_in,
                i_out=ins.i_out,
                path_shape=ins.path_shape,
                path_weight=alpha(ins)**(-0.5)
            )
            for ins in instructions
        ]

        for ins in instructions:
            if not ins.i_in < len(irreps_in):
                raise IndexError(f"{ins.i_in} is not a valid index for irreps_in")
            if not ins.i_out < len(irreps_out):
                raise IndexError(f"{ins.i_out} is not a valid index for irreps_out")
            if not (ins.i_in == -1 or irreps_in[ins.i_in].ir == irreps_out[ins.i_out].ir):
                raise ValueError(
                    f"{ins.i_in} and {ins.i_out} do not have the same irrep"
                )

        if biases is None:
            biases = len(irreps_out) * (False,)
        if isinstance(biases, bool):
            biases = [biases and ir.is_scalar() for _, ir in irreps_out]

        assert len(biases) == len(irreps_out)
        assert all(ir.is_scalar() or (not b) for b, (_, ir) in zip(biases, irreps_out))

        instructions += [
            Instruction(
                i_in=-1,
                i_out=i_out,
                path_shape=(mul_ir.dim,),
                path_weight=1.0
            )
            for i_out, (bias, mul_ir) in enumerate(zip(biases, irreps_out)) if bias
        ]

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

        self.irreps_in = irreps_in
        self.irreps_out = irreps_out
        self.instructions = instructions

        opt_defaults = e3nn.get_optimization_defaults()
        self._optimize_einsums = _optimize_einsums if _optimize_einsums is not None else opt_defaults['optimize_einsums']
        del opt_defaults

        # == Generate code ==
        graphmod, self.weight_numel, self.bias_numel = _codegen_linear(
            self.irreps_in,
            self.irreps_out,
            self.instructions,
            f_in,
            f_out,
            shared_weights=shared_weights,
            optimize_einsums=self._optimize_einsums
        )
        self._codegen_register({
            "_compiled_main": graphmod
        })

        # == Generate weights ==
        if internal_weights and self.weight_numel > 0:
            assert self.shared_weights, "Having internal weights impose shared weights"
            self.weight = torch.nn.Parameter(torch.randn(*((f_in, f_out) if f_in is not None else ()), self.weight_numel))
        else:
            # For TorchScript, there always has to be some kind of defined .weight
            self.register_buffer('weight', torch.Tensor())

        # == Generate biases ==
        if internal_weights and self.bias_numel > 0:
            assert self.shared_weights, "Having internal weights impose shared weights"
            self.bias = torch.nn.Parameter(torch.zeros(*((f_out,) if f_out is not None else ()), self.bias_numel))  # see appendix C.1 and Eq.5 of https://arxiv.org/pdf/2011.14522.pdf
        else:
            self.register_buffer('bias', torch.Tensor())

        # == Compute output mask ==
        if self.irreps_out.dim > 0:
            output_mask = torch.cat([
                torch.ones(mul_ir.dim)
                if any(
                    (ins.i_out == i_out) and (0 not in ins.path_shape)
                    for ins in self.instructions
                )
                else torch.zeros(mul_ir.dim)
                for i_out, mul_ir in enumerate(self.irreps_out)
            ])
        else:
            output_mask = torch.ones(0)
        self.register_buffer('output_mask', output_mask)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.irreps_in} -> {self.irreps_out} | {self.weight_numel} weights)"

    def forward(self, features, weight: Optional[torch.Tensor] = None, bias: Optional[torch.Tensor] = None):
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
            if self.weight_numel > 0 and not self.internal_weights:
                raise RuntimeError("Weights must be provided when internal_weights = False")
            weight = self.weight
        if bias is None:
            if self.bias_numel > 0 and not self.internal_weights:
                raise RuntimeError("Biases must be provided when internal_weights = False")
            bias = self.bias
        return self._compiled_main(features, weight, bias)

    def weight_view_for_instruction(
        self,
        instruction: int,
        weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        r"""View of weights corresponding to ``instruction``.

        Parameters
        ----------
        instruction : int
            The index of the instruction to get a view on the weights for.

        weight : `torch.Tensor`, optional
            like ``weight`` argument to ``forward()``

        Returns
        -------
        `torch.Tensor`
            A view on ``weight`` or this object's internal weights for the weights corresponding to the ``instruction`` th instruction.
        """
        if weight is None:
            assert self.internal_weights, "Weights must be provided when internal_weights = False"
            weight = self.weight
        batchshape = weight.shape[:-1]
        offset = sum(prod(ins.path_shape) for ins in self.instructions[:instruction])
        ins = self.instructions[instruction]
        return weight.narrow(-1, offset, prod(ins.path_shape)).view(batchshape + ins.path_shape)

    def weight_views(
        self,
        weight: Optional[torch.Tensor] = None,
        yield_instruction: bool = False
    ):
        r"""Iterator over weight views for all instructions.

        Parameters
        ----------
        weight : `torch.Tensor`, optional
            like ``weight`` argument to ``forward()``

        yield_instruction : `bool`, default False
            Whether to also yield the corresponding instruction.

        Yields
        ------
        If ``yield_instruction`` is ``True``, yields ``(instruction_index, instruction, weight_view)``.
        Otherwise, yields ``weight_view``.
        """
        if weight is None:
            assert self.internal_weights, "Weights must be provided when internal_weights = False"
            weight = self.weight
        batchshape = weight.shape[:-1]
        offset = 0
        for ins_i, ins in enumerate(self.instructions):
            flatsize = prod(ins.path_shape)
            this_weight = weight.narrow(-1, offset, flatsize).view(batchshape + ins.path_shape)
            offset += flatsize
            if yield_instruction:
                yield ins_i, ins, this_weight
            else:
                yield this_weight


def _codegen_linear(
    irreps_in: o3.Irreps,
    irreps_out: o3.Irreps,
    instructions: List[Instruction],
    f_in: Optional[int] = None,
    f_out: Optional[int] = None,
    shared_weights: bool = False,
    optimize_einsums: bool = True,
) -> Tuple[fx.GraphModule, int, int]:
    graph_out = fx.Graph()
    tracer_out = fx.proxy.GraphAppendingTracer(graph_out)

    # = Function definitions =
    x = fx.Proxy(graph_out.placeholder('x', torch.Tensor), tracer_out)
    ws = fx.Proxy(graph_out.placeholder('w', torch.Tensor), tracer_out)
    bs = fx.Proxy(graph_out.placeholder('b', torch.Tensor), tracer_out)

    if f_in is None:
        size = x.shape[:-1]
        outsize = size + (irreps_out.dim,)
    else:
        size = x.shape[:-2]
        outsize = size + (f_out, irreps_out.dim,)

    bias_numel = sum(irreps_out[i.i_out].dim for i in instructions if i.i_in == -1)
    if bias_numel > 0:
        if f_out is None:
            bs = bs.reshape(-1, bias_numel)
        else:
            bs = bs.reshape(-1, f_out, bias_numel)

    # = Short-circut for nothing to do =
    # We produce no code for empty instructions
    instructions = [ins for ins in instructions if 0 not in ins.path_shape]

    if len(instructions) == 0 and bias_numel == 0:
        out = x.new_zeros(outsize)

        graph_out.output(out.node, torch.Tensor)
        # Short circut
        # 0 is weight_numel
        return fx.GraphModule({}, graph_out, "linear_forward"), 0, 0

    if f_in is None:
        x = x.reshape(-1, irreps_in.dim)
    else:
        x = x.reshape(-1, f_in, irreps_in.dim)
    batch_out = x.shape[0]

    weight_numel = sum(prod(ins.path_shape) for ins in instructions if ins.i_in != -1)
    if weight_numel > 0:
        ws = ws.reshape(-1, weight_numel) if f_in is None else ws.reshape(-1, f_in, f_out, weight_numel)

    # = extract individual input irreps =
    if len(irreps_in) == 1:
        x_list = [x.reshape(batch_out, *(() if f_in is None else (f_in,)), irreps_in[0].mul, irreps_in[0].ir.dim)]
    else:
        x_list = [
            x.narrow(-1, i.start, mul_ir.dim).reshape(batch_out, *(() if f_in is None else (f_in,)), mul_ir.mul, mul_ir.ir.dim)
            for i, mul_ir in zip(irreps_in.slices(), irreps_in)
        ]

    z = '' if shared_weights else 'z'

    flat_weight_index = 0
    flat_bias_index = 0

    out_list = []

    for ins in instructions:
        mul_ir_out = irreps_out[ins.i_out]

        if ins.i_in == -1:
            # = bias =
            b = bs.narrow(-1, flat_bias_index, prod(ins.path_shape))
            flat_bias_index += prod(ins.path_shape)
            out_list += [(ins.path_weight * b).reshape(1, *(() if f_out is None else (f_out,)), mul_ir_out.dim)]
        else:
            mul_ir_in = irreps_in[ins.i_in]

            # Short-circut for empty irreps
            if mul_ir_in.dim == 0 or mul_ir_out.dim == 0:
                continue

            # Extract the weight from the flattened weight tensor
            path_nweight = prod(ins.path_shape)
            if len(instructions) == 1:
                # Avoid unnecessary view when there is only one weight
                w = ws
            else:
                w = ws.narrow(-1, flat_weight_index, path_nweight)
            w = w.reshape(
                (() if shared_weights else (-1,)) + (() if f_in is None else (f_in, f_out)) + ins.path_shape
            )
            flat_weight_index += path_nweight

            if f_in is None:
                ein_out = torch.einsum(f"{z}uw,zui->zwi", w, x_list[ins.i_in])
            else:
                ein_out = torch.einsum(f"{z}xyuw,zxui->zywi", w, x_list[ins.i_in])

            ein_out = ins.path_weight * ein_out

            out_list += [ein_out.reshape(batch_out, *(() if f_out is None else (f_out,)), mul_ir_out.dim)]

    # = Return the result =
    out = [
        _sum_tensors(
            [out for ins, out in zip(instructions, out_list) if ins.i_out == i_out],
            shape=(batch_out, *(() if f_out is None else (f_out,)), mul_ir_out.dim),
            like=x
        )
        for i_out, mul_ir_out in enumerate(irreps_out)
        if mul_ir_out.mul > 0
    ]
    if len(out) > 1:
        out = torch.cat(out, dim=-1)
    else:
        out = out[0]

    out = out.reshape(outsize)

    graph_out.output(out.node, torch.Tensor)

    # check graphs
    graph_out.lint()

    graphmod_out = fx.GraphModule({}, graph_out, "linear_forward")

    # TODO: when eliminate_dead_code() is in PyTorch stable, use that
    if optimize_einsums:
        # See _tensor_product/_codegen.py for notes
        batchdim = 4
        example_inputs = (
            torch.zeros((batchdim, *(() if f_in is None else (f_in,)), irreps_in.dim)),
            torch.zeros(
                1 if shared_weights else batchdim,
                f_in or 1,
                f_out or 1,
                weight_numel,
            ),
            torch.zeros(
                1 if shared_weights else batchdim,
                f_out or 1,
                bias_numel,
            ),
        )
        graphmod_out = optimize_einsums_full(graphmod_out, example_inputs)

    return graphmod_out, weight_numel, bias_numel
