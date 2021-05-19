from math import sqrt
from typing import List, Tuple, Union
import copy
import itertools

import torch
from torch import fx

from e3nn import o3
from e3nn.util import prod

from ._instruction import Instruction


def _sum_tensors(xs: List[torch.Tensor], shape: torch.Size, like: torch.Tensor):
    """Sum a possibly empty list of tensors with given shape"""
    if len(xs) > 0:
        out = xs[0]
        for x in xs[1:]:
            out = out + x
        return out
    return like.new_zeros(shape)


def _extract_irreps(inp, irreps):
    """Extract irreps from one tensor into a list of tensors"""
    # If only one input irrep, can avoid creating a view
    if len(irreps) == 1:
        out = [inp.reshape(-1, irreps[0].mul, irreps[0].ir.dim)]
    else:
        out = [
            inp.narrow(-1, i.start, i.stop - i.start).reshape(-1, mul_ir.mul, mul_ir.ir.dim)
            for i, mul_ir in zip(irreps.slices(), irreps)
        ]
    return out


def _combine(tensors: List[fx.Proxy], to: List[int], lengths: List[int], batch_shape, in_place: bool = False):
    """Sum/cat different tensors into a direct sum"""
    if in_place:
        node0: fx.Node = tensors[0].node
        assert isinstance(node0, fx.Node)
        graph: fx.Graph = node0.graph
        with graph.inserting_after(node0):
            bufshape = (batch_shape + (sum(lengths),))
            bufkwargs = {"device": tensors[0].device.node, "dtype": tensors[0].dtype.node}
        with graph.inserting_after(bufshape.node):
            buffer = fx.Proxy(graph.call_function(
                torch.zeros,
                args=(bufshape.node,),
                kwargs=bufkwargs
            ))
        del bufkwargs
        del bufshape
        starts = list(itertools.accumulate(lengths, initial=0))
        for i, (tensor, to_idex) in enumerate(zip(tensors, to)):
            with graph.inserting_after(buffer.node if i == 0 else tensor.node):
                bufslice = buffer.narrow(-1, starts[to_idex], lengths[to_idex])
            with graph.inserting_after(bufslice.node):
                new_shape = batch_shape + (lengths[to_idex],)
            with graph.inserting_after(new_shape.node):
                tensor_shaped = tensor.reshape(new_shape)
            with graph.inserting_after(tensor_shaped.node):
                bufslice.add_(tensor_shaped)
        return buffer
    else:
        out = [
            _sum_tensors(
                [t for i, t in zip(to, tensors) if i == i_out],
                shape=tensors[0].shape[:-1] + (lengths[i_out],),
                like=tensors[0]
            )
            for i_out in range(len(lengths))
            if lengths[i_out] > 0
        ]
        if len(out) > 1:
            out = torch.cat(out, dim=-1)
        else:
            # Avoid an unnecessary copy in a size one torch.cat
            out = out[0]
        return out.reshape(batch_shape + (sum(lengths),))


def codegen_tensor_product(
    irreps_in1: o3.Irreps,
    in1_var: List[float],
    irreps_in2: o3.Irreps,
    in2_var: List[float],
    irreps_out: o3.Irreps,
    out_var: List[float],
    instructions: List[Instruction],
    normalization: str = 'component',
    shared_weights: bool = False,
    specialized_code: bool = True,
    optimize_einsums: bool = True,
    explicit_backward: bool = False,
    instruction_profiling: bool = False
) -> Union[
    Tuple[Tuple[fx.GraphModule, fx.GraphModule], fx.GraphModule],
    Tuple[fx.GraphModule, fx.GraphModule]
]:
    graph_out = fx.Graph()
    graph_right = fx.Graph()

    # = Function definitions =
    x1_out_node = graph_out.placeholder('x1', torch.Tensor)
    x2_out_node = graph_out.placeholder('x2', torch.Tensor)
    ws_out_node = graph_out.placeholder('w', torch.Tensor)
    x1s_out = fx.Proxy(x1_out_node)
    x2s_out = fx.Proxy(x2_out_node)
    ws_out = fx.Proxy(ws_out_node)

    x2s_right = fx.Proxy(graph_right.placeholder('x2', torch.Tensor))
    ws_right = fx.Proxy(graph_right.placeholder('w', torch.Tensor))

    if not explicit_backward:
        # Explicit backward can't support broadcasting since it builds the output
        # shapes directly
        empty_out = fx.Proxy(graph_out.call_function(torch.empty, ((),), dict(device='cpu')))
        empty_right = fx.Proxy(graph_right.call_function(torch.empty, ((),), dict(device='cpu')))
        if shared_weights:
            size_out = torch.broadcast_tensors(empty_out.expand(x1s_out.shape[:-1]), empty_out.expand(x2s_out.shape[:-1]))[0].shape
            size_right = x2s_right.shape[:-1]
        else:
            size_out = torch.broadcast_tensors(empty_out.expand(x1s_out.shape[:-1]), empty_out.expand(x2s_out.shape[:-1]), empty_out.expand(ws_out.shape[:-1]))[0].shape
            size_right = torch.broadcast_tensors(empty_right.expand(x2s_right.shape[:-1]), empty_right.expand(ws_right.shape[:-1]))[0].shape
    else:
        size_out = x1s_out.shape[:-1]
        torch._assert(
            size_out == x2s_out.shape[:-1],
            "Batch shapes don't match between x1 and x2 --- please not that broadcasting is not supported with explicit_backward=True"
        )
        size_right = x2s_right.shape[:-1]

    # = Short-circut for zero dimensional =
    # We produce no code for empty instructions
    instructions = [ins for ins in instructions if 0 not in ins.path_shape]

    if len(instructions) == 0:
        out_out = x1s_out.new_zeros(size_out + (irreps_out.dim,))
        out_right = x2s_right.new_zeros(size_right + (irreps_in1.dim, irreps_out.dim,))

        graph_out.output(out_out.node, torch.Tensor)
        graph_right.output(out_right.node, torch.Tensor)
        # Short circut
        # if we short circut, no matter what, we don't do explicit backward
        # there's no reason to register it all
        return (
            fx.GraphModule({}, graph_out, "tp_forward"),
            fx.GraphModule({}, graph_right, "tp_right")
        )

    # = Broadcast inputs =
    if not explicit_backward:
        # Explicit backward can't support broadcasting since it builds the output
        # shapes directly
        if shared_weights:
            x1s_out, x2s_out = x1s_out.broadcast_to(size_out + (-1,)), x2s_out.broadcast_to(size_out + (-1,))
        else:
            x1s_out, x2s_out, ws_out = x1s_out.broadcast_to(size_out + (-1,)), x2s_out.broadcast_to(size_out + (-1,)), ws_out.broadcast_to(size_out + (-1,))
            x2s_right, ws_right = x2s_right.broadcast_to(size_right + (-1,)), ws_right.broadcast_to(size_right + (-1,))

    outsize_right = size_right + (irreps_in1.dim, irreps_out.dim,)

    x1s_out = x1s_out.reshape(-1, irreps_in1.dim)
    x2s_out = x2s_out.reshape(-1, irreps_in2.dim)
    x2s_right = x2s_right.reshape(-1, irreps_in2.dim)

    # = Determine number of weights and reshape weights ==
    weight_numel = sum(prod(ins.path_shape) for ins in instructions if ins.has_weight)
    if weight_numel > 0:
        ws_out = ws_out.reshape(-1, weight_numel)
        ws_right = ws_right.reshape(-1, weight_numel)

    # = book-keeping for wigners =
    w3j = []
    w3j_dict_out = dict()
    w3j_dict_right = dict()

    # = extract individual input irreps =
    x1_list_out = _extract_irreps(x1s_out, irreps_in1)
    x2_list_out = _extract_irreps(x2s_out, irreps_in2)
    x2_list_right = _extract_irreps(x2s_right, irreps_in2)

    # The einsum string index to prepend to the weights if the weights are not shared and have a batch dimension
    z = '' if shared_weights else 'z'

    # Cache of input irrep pairs whose outer products (xx) have already been computed
    xx_dict = dict()

    # Extract weights
    flat_weight_index = 0
    ws_list_out = []
    ws_list_right = []
    for ins in instructions:
        if ins.has_weight:
            # Extract the weight from the flattened weight tensor
            ws_list_out.append(
                ws_out.narrow(-1, flat_weight_index, prod(ins.path_shape)).reshape(
                    (() if shared_weights else (-1,)) + tuple(ins.path_shape)
                )
            )
            ws_list_right.append(
                ws_right.narrow(-1, flat_weight_index, prod(ins.path_shape)).reshape(
                    (() if shared_weights else (-1,)) + tuple(ins.path_shape)
                )
            )
            flat_weight_index += prod(ins.path_shape)
        else:
            # Keep indexing consistant
            ws_list_out.append(None)
            ws_list_right.append(None)
    del flat_weight_index

    # Lists of output proxies from each instruction
    out_list_out = []
    out_list_right = []

    for ins_idex, ins in enumerate(instructions):
        mul_ir_in1 = irreps_in1[ins.i_in1]
        mul_ir_in2 = irreps_in2[ins.i_in2]
        mul_ir_out = irreps_out[ins.i_out]

        assert mul_ir_in1.ir.p * mul_ir_in2.ir.p == mul_ir_out.ir.p
        assert abs(mul_ir_in1.ir.l - mul_ir_in2.ir.l) <= mul_ir_out.ir.l <= mul_ir_in1.ir.l + mul_ir_in2.ir.l

        if mul_ir_in1.dim == 0 or mul_ir_in2.dim == 0 or mul_ir_out.dim == 0:
            continue

        alpha = ins.path_weight * out_var[ins.i_out] / sum(in1_var[i.i_in1] * in2_var[i.i_in2] for i in instructions if i.i_out == ins.i_out)

        if instruction_profiling:
            # Open the profiler block
            name = f"{mul_ir_in1} x {mul_ir_in2} = {mul_ir_out} {ins.connection_mode} {ins.has_weight}"
            handle_out = graph_out.call_function(torch.ops.profiler._record_function_enter, (name,))
            handle_right = graph_right.call_function(torch.ops.profiler._record_function_enter, (name,))

        x1_out = x1_list_out[ins.i_in1]
        x2_out = x2_list_out[ins.i_in2]
        x2_right = x2_list_right[ins.i_in2]

        e1_right = fx.Proxy(graph_right.call_function(torch.eye, (mul_ir_in1.mul,), dict(dtype=x2s_right.dtype.node, device=x2s_right.device.node)))
        e2_right = fx.Proxy(graph_right.call_function(torch.eye, (mul_ir_in2.mul,), dict(dtype=x2s_right.dtype.node, device=x2s_right.device.node)))
        i1_right = fx.Proxy(graph_right.call_function(torch.eye, (mul_ir_in1.ir.dim,), dict(dtype=x2s_right.dtype.node, device=x2s_right.device.node)))

        assert ins.connection_mode in ['uvw', 'uvu', 'uvv', 'uuw', 'uuu', 'uvuv']

        alpha = sqrt(alpha / {
            'uvw': (mul_ir_in1.mul * mul_ir_in2.mul),
            'uvu': mul_ir_in2.mul,
            'uvv': mul_ir_in1.mul,
            'uuw': mul_ir_in1.mul,
            'uuu': 1,
            'uvuv': 1,
        }[ins.connection_mode])

        w_out = ws_list_out[ins_idex]
        w_right = ws_list_right[ins_idex]

        # Construct the general xx in case this instruction isn't specialized
        # If this isn't used, the dead code will get removed
        key = (ins.i_in1, ins.i_in2, ins.connection_mode[:2])
        if key not in xx_dict:
            if ins.connection_mode[:2] == 'uv':
                xx_dict[key] = torch.einsum('zui,zvj->zuvij', x1_out, x2_out)
            if ins.connection_mode[:2] == 'uu':
                xx_dict[key] = torch.einsum('zui,zuj->zuij', x1_out, x2_out)
        xx = xx_dict[key]

        # Create a proxy & request for the relevant wigner w3j
        # If not used (because of specialized code), will get removed later.
        key = (mul_ir_in1.ir.l, mul_ir_in2.ir.l, mul_ir_out.ir.l)
        if key not in w3j:
            w3j_dict_out[key] = fx.Proxy(graph_out.get_attr(f"_w3j_{key[0]}_{key[1]}_{key[2]}"))
            w3j_dict_right[key] = fx.Proxy(graph_right.get_attr(f"_w3j_{key[0]}_{key[1]}_{key[2]}"))
            w3j.append(key)
        w3j_out = w3j_dict_out[key]
        w3j_right = w3j_dict_right[key]

        exp = {'component': 1, 'norm': -1}[normalization]

        # - Make instruction einsum -
        if ins.connection_mode == 'uvw':
            assert ins.has_weight
            if specialized_code and key == (0, 0, 0):
                ein_out = torch.einsum(f"{z}uvw,zu,zv->zw", w_out, x1_out.reshape(-1, mul_ir_in1.dim), x2_out.reshape(-1, mul_ir_in2.dim))
                ein_right = torch.einsum(f"{z}uvw,zv->zuw", w_right, x2_right.reshape(-1, mul_ir_in2.dim))
            elif specialized_code and mul_ir_in1.ir.l == 0:
                ein_out = torch.einsum(f"{z}uvw,zu,zvj->zwj", w_out, x1_out.reshape(-1, mul_ir_in1.dim), x2_out)
                ein_right = torch.einsum(f"{z}uvw,zvi->zuwi", w_right, x2_right)
            elif specialized_code and mul_ir_in2.ir.l == 0:
                ein_out = torch.einsum(f"{z}uvw,zui,zv->zwi", w_out, x1_out, x2_out.reshape(-1, mul_ir_in2.dim))
                ein_right = torch.einsum(f"{z}uvw,ij,zv->zuiwj", w_right, i1_right, x2_right.reshape(-1, mul_ir_in2.dim))
            elif specialized_code and mul_ir_out.ir.l == 0:
                ein_out = torch.einsum(f"{z}uvw,zui,zvi->zw", w_out, x1_out, x2_out) / sqrt(mul_ir_in1.ir.dim)**exp
                ein_right = torch.einsum(f"{z}uvw,zvi->zuiw", w_right, x2_right) / sqrt(mul_ir_in1.ir.dim)**exp
            else:
                ein_out = torch.einsum(f"{z}uvw,ijk,zuvij->zwk", w_out, w3j_out, xx)
                ein_right = torch.einsum(f"{z}uvw,ijk,zvj->zuiwk", w_right, w3j_right, x2_right)
        if ins.connection_mode == 'uvu':
            assert mul_ir_in1.mul == mul_ir_out.mul
            if ins.has_weight:
                if specialized_code and key == (0, 0, 0):
                    ein_out = torch.einsum(f"{z}uv,zu,zv->zu", w_out, x1_out.reshape(-1, mul_ir_in1.dim), x2_out.reshape(-1, mul_ir_in2.dim))
                    ein_right = torch.einsum(f"{z}uv,uw,zv->zuw", w_right, e1_right, x2_right.reshape(-1, mul_ir_in2.dim))
                elif specialized_code and mul_ir_in1.ir.l == 0:
                    ein_out = torch.einsum(f"{z}uv,zu,zvj->zuj", w_out, x1_out.reshape(-1, mul_ir_in1.dim), x2_out)
                    ein_right = torch.einsum(f"{z}uv,uw,zvi->zuwi", w_right, e1_right, x2_right)
                elif specialized_code and mul_ir_in2.ir.l == 0:
                    ein_out = torch.einsum(f"{z}uv,zui,zv->zui", w_out, x1_out, x2_out.reshape(-1, mul_ir_in2.dim))
                    ein_right = torch.einsum(f"{z}uv,ij,uw,zv->zuiwj", w_right, i1_right, e1_right, x2_right.reshape(-1, mul_ir_in2.dim))
                elif specialized_code and mul_ir_out.ir.l == 0:
                    ein_out = torch.einsum(f"{z}uv,zui,zvi->zu", w_out, x1_out, x2_out) / sqrt(mul_ir_in1.ir.dim)**exp
                    ein_right = torch.einsum(f"{z}uv,uw,zvi->zuiw", w_right, e1_right, x2_right) / sqrt(mul_ir_in1.ir.dim)**exp
                else:
                    ein_out = torch.einsum(f"{z}uv,ijk,zuvij->zuk", w_out, w3j_out, xx)
                    ein_right = torch.einsum(f"{z}uv,ijk,uw,zvj->zuiwk", w_right, w3j_right, e1_right, x2_right)
            else:
                # not so useful operation because v is summed
                ein_out = torch.einsum("ijk,zuvij->zuk", w3j_out, xx)
                ein_right = torch.einsum("ijk,uw,zvj->zuiwk", w3j_right, e1_right, x2_right)
        if ins.connection_mode == 'uvv':
            assert mul_ir_in2.mul == mul_ir_out.mul
            if ins.has_weight:
                if specialized_code and key == (0, 0, 0):
                    ein_out = torch.einsum(f"{z}uv,zu,zv->zv", w_out, x1_out.reshape(-1, mul_ir_in1.dim), x2_out.reshape(-1, mul_ir_in2.dim))
                    ein_right = torch.einsum(f"{z}uv,vw,zv->zuw", w_right, e2_right, x2_right.reshape(-1, mul_ir_in2.dim))
                elif specialized_code and mul_ir_in1.ir.l == 0:
                    ein_out = torch.einsum(f"{z}uv,zu,zvj->zvj", w_out, x1_out.reshape(-1, mul_ir_in1.dim), x2_out)
                    ein_right = torch.einsum(f"{z}uv,vw,zvi->zuwi", w_right, e2_right, x2_right)
                elif specialized_code and mul_ir_in2.ir.l == 0:
                    ein_out = torch.einsum(f"{z}uv,zui,zv->zvi", w_out, x1_out, x2_out.reshape(-1, mul_ir_in2.dim))
                    ein_right = torch.einsum(f"{z}uv,ij,vw,zv->zuiwj", w_right, i1_right, e2_right, x2_right.reshape(-1, mul_ir_in2.dim))
                elif specialized_code and mul_ir_out.ir.l == 0:
                    ein_out = torch.einsum(f"{z}uv,zui,zvi->zv", w_out, x1_out, x2_out) / sqrt(mul_ir_in1.ir.dim)**exp
                    ein_right = torch.einsum(f"{z}uv,vw,zvi->zuiw", w_right, e2_right, x2_right) / sqrt(mul_ir_in1.ir.dim)**exp
                else:
                    ein_out = torch.einsum(f"{z}uv,ijk,zuvij->zvk", w_out, w3j_out, xx)
                    ein_right = torch.einsum(f"{z}uv,ijk,zvj->zuivk", w_right, w3j_right, x2_right)
            else:
                # not so useful operation because u is summed
                # only specialize out for this path
                if specialized_code and key == (0, 0, 0):
                    ein_out = torch.einsum("zu,zv->zv", x1_out.reshape(-1, mul_ir_in1.dim), x2_out.reshape(-1, mul_ir_in2.dim))
                elif specialized_code and mul_ir_in1.ir.l == 0:
                    ein_out = torch.einsum("zu,zvj->zvj", x1_out.reshape(-1, mul_ir_in1.dim), x2_out)
                elif specialized_code and mul_ir_in2.ir.l == 0:
                    ein_out = torch.einsum("zui,zv->zvi", x1_out, x2_out.reshape(-1, mul_ir_in2.dim))
                elif specialized_code and mul_ir_out.ir.l == 0:
                    ein_out = torch.einsum("zui,zvi->zv", x1_out, x2_out) / sqrt(mul_ir_in1.ir.dim)**exp
                else:
                    ein_out = torch.einsum("ijk,zuvij->zvk", w3j_out, xx)
                s2ones = fx.Proxy(graph_right.call_function(torch.ones, (mul_ir_in1.mul,), dict(device=x2_right.device.node, dtype=x2_right.dtype.node)))
                ein_right = torch.einsum("u,ijk,zvj->zuivk", s2ones, w3j_right, x2_right)
        if ins.connection_mode == 'uuw':
            assert mul_ir_in1.mul == mul_ir_in2.mul
            if ins.has_weight:
                if specialized_code and key == (0, 0, 0):
                    ein_out = torch.einsum(f"{z}uw,zu,zu->zw", w_out, x1_out.reshape(-1, mul_ir_in1.dim), x2_out.reshape(-1, mul_ir_in2.dim))
                elif specialized_code and mul_ir_in1.ir.l == 0:
                    ein_out = torch.einsum(f"{z}uw,zu,zuj->zwj", w_out, x1_out.reshape(-1, mul_ir_in1.dim), x2_out)
                elif specialized_code and mul_ir_in2.ir.l == 0:
                    ein_out = torch.einsum(f"{z}uw,zui,zu->zwi", w_out, x1_out, x2_out.reshape(-1, mul_ir_in2.dim))
                elif specialized_code and mul_ir_out.ir.l == 0:
                    ein_out = torch.einsum(f"{z}uw,zui,zui->zw", w_out, x1_out, x2_out) / sqrt(mul_ir_in1.ir.dim)**exp
                else:
                    ein_out = torch.einsum(f"{z}uw,ijk,zuij->zwk", w_out, w3j_out, xx)
                # TODO: specialize right()
                ein_right = torch.einsum(f"{z}uw,ijk,zuj->zuiwk", w_right, w3j_right, x2_right)
            else:
                # equivalent to tp(x, y, 'uuu').sum('u')
                assert mul_ir_out.mul == 1
                ein_out = torch.einsum("ijk,zuij->zk", w3j_out, xx)
                ein_right = torch.einsum("ijk,zuj->zuik", w3j_right, x2_right)
        if ins.connection_mode == 'uuu':
            assert mul_ir_in1.mul == mul_ir_in2.mul == mul_ir_out.mul
            if ins.has_weight:
                if specialized_code and key == (0, 0, 0):
                    ein_out = torch.einsum(f"{z}u,zu,zu->zu", w_out, x1_out.reshape(-1, mul_ir_in1.dim), x2_out.reshape(-1, mul_ir_in2.dim))
                    ein_right = torch.einsum(f"{z}u,uw,zu->zuw", w_right, e2_right, x2_right.reshape(-1, mul_ir_in2.dim))
                # TODO: support cross!
                # elif specialized_code and key == (1, 1, 1) and normalization == "component":
                #     ein_out = torch.einsum(
                #         f"{z}u,zui->zui",
                #         w_out,
                #         torch.cross(x1_out, x2_out, dim=2)
                #     ) / sqrt(2)
                #     # For cross product, use the general case right()
                #     ein_right = torch.einsum(f"{z}u,ijk,uw,zuj->zuiwk", w_right, w3j_right, e1_right, x2_right)
                elif specialized_code and mul_ir_in1.ir.l == 0:
                    ein_out = torch.einsum(f"{z}u,zu,zuj->zuj", w_out, x1_out.reshape(-1, mul_ir_in1.dim), x2_out)
                    ein_right = torch.einsum(f"{z}u,uw,zui->zuwi", w_right, e2_right, x2_right)
                elif specialized_code and mul_ir_in2.ir.l == 0:
                    ein_out = torch.einsum(f"{z}u,zui,zu->zui", w_out, x1_out, x2_out.reshape(-1, mul_ir_in2.dim))
                    ein_right = torch.einsum(f"{z}u,ij,uw,zu->zuiwj", w_right, i1_right, e2_right, x2_right.reshape(-1, mul_ir_in2.dim))
                elif specialized_code and mul_ir_out.ir.l == 0:
                    ein_out = torch.einsum(f"{z}u,zui,zui->zu", w_out, x1_out, x2_out) / sqrt(mul_ir_in1.ir.dim)**exp
                    ein_right = torch.einsum(f"{z}u,uw,zui->zuiw", w_right, e2_right, x2_right) / sqrt(mul_ir_in1.ir.dim)**exp
                else:
                    ein_out = torch.einsum(f"{z}u,ijk,zuij->zuk", w_out, w3j_out, xx)
                    ein_right = torch.einsum(f"{z}u,ijk,uw,zuj->zuiwk", w_right, w3j_right, e1_right, x2_right)
            else:
                if specialized_code and key == (0, 0, 0):
                    ein_out = torch.einsum("zu,zu->zu", x1_out.reshape(-1, mul_ir_in1.dim), x2_out.reshape(-1, mul_ir_in2.dim))
                    ein_right = torch.einsum("uw,zu->zuw", e2_right, x2_right.reshape(-1, mul_ir_in2.dim))
                # elif specialized_code and key == (1, 1, 1) and normalization == "component":
                #     ein_out = torch.cross(x1_out, x2_out, dim=2) * (1.0 / sqrt(2))
                #     # For cross product, use the general case right()
                #     ein_right = torch.einsum("ijk,uw,zuj->zuiwk", w3j_right, e1_right, x2_right)
                elif specialized_code and mul_ir_in1.ir.l == 0:
                    ein_out = torch.einsum("zu,zuj->zuj", x1_out.reshape(-1, mul_ir_in1.dim), x2_out)
                    ein_right = torch.einsum("uw,zui->zuwi", e2_right, x2_right)
                elif specialized_code and mul_ir_in2.ir.l == 0:
                    ein_out = torch.einsum("zui,zu->zui", x1_out, x2_out.reshape(-1, mul_ir_in2.dim))
                    ein_right = torch.einsum("ij,uw,zu->zuiwj", i1_right, e2_right, x2_right.reshape(-1, mul_ir_in2.dim))
                elif specialized_code and mul_ir_out.ir.l == 0:
                    ein_out = torch.einsum("zui,zui->zu", x1_out, x2_out) / sqrt(mul_ir_in1.ir.dim)**exp
                    ein_right = torch.einsum("uw,zui->zuiw", e2_right, x2_right) / sqrt(mul_ir_in1.ir.dim)**exp
                else:
                    ein_out = torch.einsum("ijk,zuij->zuk", w3j_out, xx)
                    ein_right = torch.einsum("ijk,uw,zuj->zuiwk", w3j_right, e1_right, x2_right)
        if ins.connection_mode == 'uvuv':
            assert mul_ir_in1.mul * mul_ir_in2.mul == mul_ir_out.mul
            if ins.has_weight:
                # TODO implement specialized code
                ein_out = torch.einsum(f"{z}uv,ijk,zuvij->zuvk", w_out, w3j_out, xx)
                ein_right = torch.einsum(f"{z}uv,ijk,uw,zvj->zuiwvk", w_right, w3j_right, e1_right, x2_right)
            else:
                # TODO implement specialized code
                ein_out = torch.einsum("ijk,zuvij->zuvk", w3j_out, xx)
                ein_right = torch.einsum("ijk,uw,zvj->zuiwvk", w3j_right, e1_right, x2_right)
        # - end make einsum -

        ein_out = alpha * ein_out
        ein_right = alpha * ein_right

        out_list_out.append(ein_out.reshape(-1, mul_ir_out.dim))
        out_list_right.append(ein_right.reshape(-1, mul_ir_in1.dim, mul_ir_out.dim))

        if instruction_profiling:
            # Close the profiler block
            graph_out.call_function(torch.ops.profiler._record_function_exit, (handle_out,))
            graph_right.call_function(torch.ops.profiler._record_function_exit, (handle_right,))

        # Remove unused w3js:
        if len(w3j_out.node.users) == 0 and len(w3j_right.node.users) == 0:
            del w3j[-1]
            # The w3j nodes are reshapes, so we have to remove them from the graph
            # Although they are dead code, they try to reshape to dimensions that don't exist
            # (since the corresponding w3js are not in w3j)
            # so they screw up the shape propagation, even though they would be removed later as dead code by TorchScript.
            graph_out.erase_node(w3j_dict_out.pop(key).node)
            graph_right.erase_node(w3j_dict_right.pop(key).node)
    # - end make instructions -

    # === Make a graph for backward ===
    if explicit_backward:
        # We create a graph here so that it doesn't have the output stage for forward
        graph_backward = copy.deepcopy(graph_out)

    # === Output for forward ===
    out_out = _combine(
        tensors=out_list_out,
        to=[ins.i_out for ins in instructions],
        lengths=[mul_ir.dim for mul_ir in irreps_out],
        batch_shape=size_out,
        # If we are doing explicit backward, the inside of forward isn't seen by autograd
        # As a result, we want to use as many in-place operations as possible to save memory
        in_place=explicit_backward
    )

    out_right = [
        torch.cat([
            _sum_tensors(
                [out for ins, out in zip(instructions, out_list_right) if (ins.i_in1, ins.i_out) == (i_in1, i_out)],
                shape=(x2s_right.shape[0], mul_ir_in1.dim, mul_ir_out.dim),
                like=x2s_right
            )
            for i_out, mul_ir_out in enumerate(irreps_out)
            if mul_ir_out.mul > 0
        ], dim=2)
        for i_in1, mul_ir_in1 in enumerate(irreps_in1)
        if mul_ir_in1.mul > 0
    ]
    if len(out_right) > 1:
        out_right = torch.cat(out_right, dim=1)
    else:
        out_right = out_right[0]

    out_right = out_right.reshape(outsize_right)

    if explicit_backward:
        # We also have to output the tensors we want to save for the backward pass
        # Currently, thats the inputs as well as any used outer products
        saved_xxs = {k: proxy.node for k, proxy in xx_dict.items() if len(proxy.node.users) > 0}
        graph_out.output(
            (
                out_out.node,
                # what's to save for backward
                (x1_out_node, x2_out_node, ws_out_node,) + tuple(saved_xxs.values())
            ),
            Tuple[torch.Tensor, Tuple[(torch.Tensor,)*(len(saved_xxs) + 3)]]
        )
    else:
        # The normal case: just return the output
        graph_out.output(out_out.node, torch.Tensor)
    graph_right.output(out_right.node, torch.Tensor)

    # === Check graphs ===
    graph_out.lint()
    graph_right.lint()

    # === Make GraphModules ===
    wigner_mats = {}
    for l_1, l_2, l_out in w3j:
        wig = o3.wigner_3j(l_1, l_2, l_out)

        if normalization == 'component':
            wig *= (2 * l_out + 1) ** 0.5
        if normalization == 'norm':
            wig *= (2 * l_1 + 1) ** 0.5 * (2 * l_2 + 1) ** 0.5

        wigner_mats[f"_w3j_{l_1}_{l_2}_{l_out}"] = wig

    graphmod_out = fx.GraphModule(wigner_mats, graph_out, class_name="tp_forward")
    if explicit_backward:
        graphmod_backward = fx.GraphModule(wigner_mats, graph_backward, class_name="tp_backward")
    graphmod_right = fx.GraphModule(wigner_mats, graph_right, class_name="tp_right")

    # ===== Build backward =====
    if explicit_backward:
        try:
            from opt_einsum_fx.grad import grad
        except ImportError:
            raise ImportError("opt_einsum_fx is required for explicit_backward = True")

        # Gradients with reshapes need shape info
        # We run shape propagation now while the graph still only has inputs that are easy to know the shapes of
        from torch.fx.passes.shape_prop import ShapeProp
        sp = ShapeProp(graphmod_backward)
        sp.run(
            torch.zeros(1, irreps_in1.dim),
            torch.zeros(1, irreps_in2.dim),
            torch.zeros(((1,) if shared_weights else tuple()) + (weight_numel,)),
        )

        # - Find certain nodes in the grad graph -
        # There doesn't seem to be a better way to identify nodes across a graph copy
        # since fx.Graph removes all custom Node attributes during copy
        def find_in_graph_copy(graph: fx.Graph, nodes: List[Union[fx.Node, fx.Proxy]]) -> List[fx.Node]:
            """ !! NOT a general function --- only works if the graphs are copies."""
            nodes = [n.node if isinstance(n, fx.Proxy) else n for n in nodes]
            found = {None: None}
            ids = [str(n) if n is not None else None for n in nodes]
            for node in graph.nodes:
                node_id = str(node)
                if node_id in ids:
                    found[node_id] = node
            return [found[node_id] for node_id in ids]

        grad_x1_inputs = find_in_graph_copy(graph_backward, x1_list_out)
        grad_x2_inputs = find_in_graph_copy(graph_backward, x2_list_out)
        grad_ws_inputs = find_in_graph_copy(graph_backward, ws_list_out)
        grad_out_list = find_in_graph_copy(graph_backward, out_list_out)
        grad_size_out = fx.Proxy(find_in_graph_copy(graph_backward, [size_out.node])[0])
        # We need to find the outer products that we might be able to use from the cache
        # But we leave them for now since they have to be present for grad()
        grad_xx_dict = dict(zip(
            saved_xxs.keys(),
            find_in_graph_copy(graph_backward, list(saved_xxs.values()))
        ))
        grad_xx_shapes = [n.shape[1:] for n in grad_xx_dict.values()]
        # - Add placeholders for xx saved tensors -
        # Placeholders for the outer products cached from forward
        grad_xx_placeholders = []
        for xx_key, xx_node in grad_xx_dict.items():
            # Use x1 to get these in before grad_out for correct order
            with graph_backward.inserting_before(grad_x1_inputs[0]):
                grad_xx_placeholders.append(graph_backward.placeholder(f"xx_{xx_key[0]}_{xx_key[1]}_{xx_key[2]}", torch.Tensor))

        # - Add a gradient input to the backward graph -
        # insert before x2 extract to make sure its after the xx placeholders to maintain correct parameter order
        with graph_backward.inserting_before(grad_x2_inputs[0]):
            grad_out = fx.Proxy(graph_backward.placeholder('grad_out', torch.Tensor))
            grad_out = grad_out.reshape(-1, irreps_out.dim)
            grad_grad_list = _extract_irreps(grad_out, irreps_out)
            grad_grad_list = [p.node for p in grad_grad_list]
        del grad_out

        # - Compute symbolic gradients -
        # Get gradient graphs
        grad_x1s = []
        grad_x2s = []
        grad_ws = []
        for ins_i, (ins, node) in enumerate(zip(instructions, grad_out_list)):
            # grads wrt all three vars
            grad_x1s.append(grad(node, grad_grad_list[ins.i_out], grad_x1_inputs[ins.i_in1]))
            grad_x2s.append(grad(node, grad_grad_list[ins.i_out], grad_x2_inputs[ins.i_in2]))
            if ins.has_weight:
                grad_ws.append(grad(node, grad_grad_list[ins.i_out], grad_ws_inputs[ins_i]))
            else:
                grad_ws.append(None)
        grad_x1s = [fx.Proxy(n) for n in grad_x1s]
        grad_x2s = [fx.Proxy(n) for n in grad_x2s]
        grad_ws = [fx.Proxy(n) if n is not None else None for n in grad_ws]

        # - Use precomputed xx outer products -
        # Now that we've computed gradients, we can replace any remaining outer products with their cached versions:
        for (xx_key, xx_node), xx_inp in zip(grad_xx_dict.items(), grad_xx_placeholders):
            xx_node.replace_all_uses_with(xx_inp)
            graph_backward.erase_node(xx_node)
        del grad_xx_dict
        del grad_xx_placeholders

        # - Build output gradients -
        # This works because gradient is a linear operation => we can interchange it across the final sums + cats.
        grad_x1s = [g.reshape(-1, irreps_in1[ins.i_in1].dim) for g, ins in zip(grad_x1s, instructions)]
        grad_x1s = _combine(
            grad_x1s,
            to=[ins.i_in1 for ins in instructions],
            lengths=[mul_ir.dim for mul_ir in irreps_in1],
            batch_shape=grad_size_out
        )
        grad_x2s = [g.reshape(-1, irreps_in2[ins.i_in2].dim) for g, ins in zip(grad_x2s, instructions)]
        grad_x2s = _combine(
            grad_x2s,
            to=[ins.i_in2 for ins in instructions],
            lengths=[mul_ir.dim for mul_ir in irreps_in2],
            batch_shape=grad_size_out
        )
        grad_ws = [gw.reshape(-1, prod(ins.path_shape)) for gw, ins in zip(grad_ws, instructions) if gw is not None]
        if len(grad_ws) > 0:
            grad_ws = _combine(
                grad_ws,
                to=range(len(grad_ws)),
                lengths=[prod(ins.path_shape) for ins in instructions if ins.has_weight], batch_shape=tuple() if shared_weights else grad_size_out
            )
        else:
            grad_ws = None

        # having make a GraphModule previously seems to add a None output automatically
        # we need to remove this so that the function doesn't return early
        for node in graph_backward.nodes:
            if node.op == "output":
                graph_backward.erase_node(node)

        graph_backward.output(
            (grad_x1s.node, grad_x2s.node, grad_ws.node if grad_ws is not None else None),
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        )

        graph_backward.lint()

        graphmod_backward.graph = graph_backward
        graphmod_backward.recompile()
    # ==== end build backward ====

    # == Eliminate dead code ==
    from opt_einsum_fx import fuse_reshapes
    from opt_einsum_fx.fx_utils import eliminate_dead_code
    for gm in (graphmod_out, graphmod_right) + (
        (graphmod_backward,) if explicit_backward
        else tuple()
    ):
        eliminate_dead_code(gm.graph)
        fuse_reshapes(gm.graph, in_place=True)
        gm.recompile()

    # == Optimize ==
    # TODO: when eliminate_dead_code() is in PyTorch stable, use that
    if optimize_einsums:
        try:
            from opt_einsum_fx import optimize_einsums_full, jitable
            from opt_einsum_fx.fx_utils import deduplicate
        except ImportError:
            # opt_einsum_fx is not installed
            pass
        else:
            # Note that for our einsums, we can optimize _once_ for _any_ batch dimension
            # and still get the right path for _all_ batch dimensions.
            # This is because our einsums are essentially of the form:
            #    zuvw,ijk,zuvij->zwk    OR     uvw,ijk,zuvij->zwk
            # In the first case, all but one operands have the batch dimension
            #    => The first contraction gains the batch dimension
            #    => All following contractions have batch dimension
            #    => All possible contraction paths have cost that scales linearly in batch size
            #    => The optimal path is the same for all batch sizes
            # For the second case, this logic follows as long as the first contraction is not between the first two operands. Since those two operands do not share any indexes, contracting them first is a rare pathological case. See
            # https://github.com/dgasmith/opt_einsum/issues/158
            # for more details.
            #
            # TODO: consider the impact maximum intermediate result size on this logic
            #         \- this is the `memory_limit` option in opt_einsum
            # TODO: allow user to choose opt_einsum parameters?
            #
            batchdim = 4
            example_inputs = (
                torch.zeros((batchdim, irreps_in1.dim)),
                torch.zeros((batchdim, irreps_in2.dim)),
                torch.zeros(
                    1 if shared_weights else batchdim,
                    weight_numel
                ),
            )

            graphmod_out = jitable(optimize_einsums_full(
                graphmod_out,
                example_inputs,
                # if explict backward, we want to do as much as possible in-place
                in_place_muls=explicit_backward
            ))
            deduplicate(graphmod_out.graph)
            graphmod_out.recompile()

            graphmod_right = jitable(optimize_einsums_full(graphmod_right, example_inputs[1:]))
            deduplicate(graphmod_right.graph)
            graphmod_out.recompile()

            if explicit_backward:
                graphmod_backward = jitable(optimize_einsums_full(
                    graphmod_backward,
                    example_inputs + tuple(
                        torch.zeros((batchdim,) + gs)  # from the shapeprop
                        for gs in grad_xx_shapes
                    ) + (
                        # grad_out
                        torch.ones(batchdim, irreps_out.dim),
                    )
                ))
                deduplicate(graphmod_backward.graph)
                graphmod_backward.recompile()

    if explicit_backward:
        return (graphmod_out, graphmod_backward), graphmod_right
    else:
        return graphmod_out, graphmod_right
