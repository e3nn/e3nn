from math import sqrt
from typing import List, Tuple

import torch
from e3nn import o3
from e3nn.util import prod
from opt_einsum_fx import jitable, optimize_einsums_full
from torch import fx

from ._instruction import Instruction


def _sum_tensors(xs: List[torch.Tensor], shape: torch.Size, like: torch.Tensor):
    if len(xs) > 0:
        out = xs[0]
        for x in xs[1:]:
            out = out + x
        return out
    return like.new_zeros(shape)


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
) -> Tuple[fx.Graph, fx.Graph, list]:
    graph_out = fx.Graph()
    graph_right = fx.Graph()

    # = Function definitions =
    x1s_out = fx.Proxy(graph_out.placeholder('x1', torch.Tensor))
    x2s_out = fx.Proxy(graph_out.placeholder('x2', torch.Tensor))
    ws_out = fx.Proxy(graph_out.placeholder('w', torch.Tensor))
    w3js_out = fx.Proxy(graph_out.placeholder('w3j', torch.Tensor))

    x2s_right = fx.Proxy(graph_right.placeholder('x2', torch.Tensor))
    ws_right = fx.Proxy(graph_right.placeholder('w', torch.Tensor))
    w3js_right = fx.Proxy(graph_right.placeholder('w3j', torch.Tensor))

    empty_out = fx.Proxy(graph_out.call_function(torch.empty, ((),), dict(device='cpu')))
    empty_right = fx.Proxy(graph_right.call_function(torch.empty, ((),), dict(device='cpu')))
    if shared_weights:
        size_out = torch.broadcast_tensors(empty_out.expand(x1s_out.shape[:-1]), empty_out.expand(x2s_out.shape[:-1]))[0].shape
        size_right = x2s_right.shape[:-1]
    else:
        size_out = torch.broadcast_tensors(empty_out.expand(x1s_out.shape[:-1]), empty_out.expand(x2s_out.shape[:-1]), empty_out.expand(ws_out.shape[:-1]))[0].shape
        size_right = torch.broadcast_tensors(empty_right.expand(x2s_right.shape[:-1]), empty_right.expand(ws_right.shape[:-1]))[0].shape

    # = Short-circut for zero dimensional =
    # We produce no code for empty instructions
    instructions = [ins for ins in instructions if 0 not in ins.path_shape]

    if len(instructions) == 0:
        out_out = x1s_out.new_zeros(size_out + (irreps_out.dim,))
        out_right = x2s_right.new_zeros(size_right + (irreps_in1.dim, irreps_out.dim,))

        graph_out.output(out_out.node, torch.Tensor)
        graph_right.output(out_right.node, torch.Tensor)
        # Short circut
        # the empty list is wigners
        return graph_out, graph_right, []

    # = Broadcast inputs =
    if shared_weights:
        x1s_out, x2s_out = x1s_out.broadcast_to(size_out + (-1,)), x2s_out.broadcast_to(size_out + (-1,))
    else:
        x1s_out, x2s_out, ws_out = x1s_out.broadcast_to(size_out + (-1,)), x2s_out.broadcast_to(size_out + (-1,)), ws_out.broadcast_to(size_out + (-1,))
        x2s_right, ws_right = x2s_right.broadcast_to(size_right + (-1,)), ws_right.broadcast_to(size_right + (-1,))

    outsize_out = size_out + (irreps_out.dim,)
    outsize_right = size_right + (irreps_in1.dim, irreps_out.dim,)

    x1s_out = x1s_out.reshape(-1, irreps_in1.dim)
    x2s_out = x2s_out.reshape(-1, irreps_in2.dim)
    x2s_right = x2s_right.reshape(-1, irreps_in2.dim)

    batch_out = x1s_out.shape[0]
    batch_right = x2s_right.shape[0]

    # = Determine number of weights and reshape weights ==
    weight_numel = sum(prod(ins.path_shape) for ins in instructions if ins.has_weight)
    if weight_numel > 0:
        ws_out = ws_out.reshape(-1, weight_numel)
        ws_right = ws_right.reshape(-1, weight_numel)
    del weight_numel

    # = book-keeping for wigners =
    w3j = []
    w3j_dict_out = dict()
    w3j_dict_right = dict()

    def w3j_dim(l1, l2, l3):
        return (2 * l1 + 1) * (2 * l2 + 1) * (2 * l3 + 1)

    # = extract individual input irreps =
    # If only one input irrep, can avoid creating a view
    if len(irreps_in1) == 1:
        x1_list_out = [x1s_out.reshape(batch_out, irreps_in1[0].mul, irreps_in1[0].ir.dim)]
    else:
        x1_list_out = [
            x1s_out[:, i].reshape(batch_out, mul_ir.mul, mul_ir.ir.dim)
            for i, mul_ir in zip(irreps_in1.slices(), irreps_in1)
        ]

    x2_list_out = []
    x2_list_right = []
    # If only one input irrep, can avoid creating a view
    if len(irreps_in2) == 1:
        x2_list_out.append(
            x2s_out.reshape(batch_out, irreps_in2[0].mul, irreps_in2[0].ir.dim)
        )
        x2_list_right.append(
            x2s_right.reshape(batch_right, irreps_in2[0].mul, irreps_in2[0].ir.dim)
        )
    else:
        for i, mul_ir in zip(irreps_in2.slices(), irreps_in2):
            x2_list_out.append(
                x2s_out[:, i].reshape(batch_out, mul_ir.mul, mul_ir.ir.dim)
            )
            x2_list_right.append(
                x2s_right[:, i].reshape(batch_right, mul_ir.mul, mul_ir.ir.dim)
            )

    # The einsum string index to prepend to the weights if the weights are not shared and have a batch dimension
    z = '' if shared_weights else 'z'

    # Cache of input irrep pairs whose outer products (xx) have already been computed
    xx_dict = dict()

    # Current index in the flat weight tensor
    flat_weight_index = 0

    out_list_out = []
    out_list_right = []

    for ins in instructions:
        mul_ir_in1 = irreps_in1[ins.i_in1]
        mul_ir_in2 = irreps_in2[ins.i_in2]
        mul_ir_out = irreps_out[ins.i_out]

        assert mul_ir_in1.ir.p * mul_ir_in2.ir.p == mul_ir_out.ir.p
        assert abs(mul_ir_in1.ir.l - mul_ir_in2.ir.l) <= mul_ir_out.ir.l <= mul_ir_in1.ir.l + mul_ir_in2.ir.l

        if mul_ir_in1.dim == 0 or mul_ir_in2.dim == 0 or mul_ir_out.dim == 0:
            continue

        alpha = ins.path_weight * out_var[ins.i_out] / sum(in1_var[i.i_in1] * in2_var[i.i_in2] for i in instructions if i.i_out == ins.i_out)

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

        if ins.has_weight:
            # Extract the weight from the flattened weight tensor
            w_out = ws_out[:, flat_weight_index:flat_weight_index + prod(ins.path_shape)].reshape((() if shared_weights else (-1,)) + tuple(ins.path_shape))
            w_right = ws_right[:, flat_weight_index:flat_weight_index + prod(ins.path_shape)].reshape((() if shared_weights else (-1,)) + tuple(ins.path_shape))
            flat_weight_index += prod(ins.path_shape)

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
            i = sum(w3j_dim(*k) for k in w3j)
            w3j_dict_out[key] = w3js_out[i:i + w3j_dim(*key)].reshape(mul_ir_in1.ir.dim, mul_ir_in2.ir.dim, mul_ir_out.ir.dim)
            w3j_dict_right[key] = w3js_right[i:i + w3j_dim(*key)].reshape(mul_ir_in1.ir.dim, mul_ir_in2.ir.dim, mul_ir_out.ir.dim)
            w3j.append(key)
        w3j_out = w3j_dict_out[key]
        w3j_right = w3j_dict_right[key]

        exp = {'component': 1, 'norm': -1}[normalization]

        if ins.connection_mode == 'uvw':
            assert ins.has_weight
            if specialized_code and key == (0, 0, 0):
                ein_out = torch.einsum(f"{z}uvw,zu,zv->zw", w_out, x1_out.reshape(batch_out, mul_ir_in1.dim), x2_out.reshape(batch_out, mul_ir_in2.dim))
                ein_right = torch.einsum(f"{z}uvw,zv->zuw", w_right, x2_right.reshape(batch_right, mul_ir_in2.dim))
            elif specialized_code and mul_ir_in1.ir.l == 0:
                ein_out = torch.einsum(f"{z}uvw,zu,zvj->zwj", w_out, x1_out.reshape(batch_out, mul_ir_in1.dim), x2_out)
                ein_right = torch.einsum(f"{z}uvw,zvi->zuwi", w_right, x2_right)
            elif specialized_code and mul_ir_in2.ir.l == 0:
                ein_out = torch.einsum(f"{z}uvw,zui,zv->zwi", w_out, x1_out, x2_out.reshape(batch_out, mul_ir_in2.dim))
                ein_right = torch.einsum(f"{z}uvw,ij,zv->zuiwj", w_right, i1_right, x2_right.reshape(batch_right, mul_ir_in2.dim))
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
                    ein_out = torch.einsum(f"{z}uv,zu,zv->zu", w_out, x1_out.reshape(batch_out, mul_ir_in1.dim), x2_out.reshape(batch_out, mul_ir_in2.dim))
                    ein_right = torch.einsum(f"{z}uv,uw,zv->zuw", w_right, e1_right, x2_right.reshape(batch_right, mul_ir_in2.dim))
                elif specialized_code and mul_ir_in1.ir.l == 0:
                    ein_out = torch.einsum(f"{z}uv,zu,zvj->zuj", w_out, x1_out.reshape(batch_out, mul_ir_in1.dim), x2_out)
                    ein_right = torch.einsum(f"{z}uv,uw,zvi->zuwi", w_right, e1_right, x2_right)
                elif specialized_code and mul_ir_in2.ir.l == 0:
                    ein_out = torch.einsum(f"{z}uv,zui,zv->zui", w_out, x1_out, x2_out.reshape(batch_out, mul_ir_in2.dim))
                    ein_right = torch.einsum(f"{z}uv,ij,uw,zv->zuiwj", w_right, i1_right, e1_right, x2_right.reshape(batch_right, mul_ir_in2.dim))
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
                    ein_out = torch.einsum(f"{z}uv,zu,zv->zv", w_out, x1_out.reshape(batch_out, mul_ir_in1.dim), x2_out.reshape(batch_out, mul_ir_in2.dim))
                    ein_right = torch.einsum(f"{z}uv,vw,zv->zuw", w_right, e2_right, x2_right.reshape(batch_right, mul_ir_in2.dim))
                elif specialized_code and mul_ir_in1.ir.l == 0:
                    ein_out = torch.einsum(f"{z}uv,zu,zvj->zvj", w_out, x1_out.reshape(batch_out, mul_ir_in1.dim), x2_out)
                    ein_right = torch.einsum(f"{z}uv,vw,zvi->zuwi", w_right, e2_right, x2_right)
                elif specialized_code and mul_ir_in2.ir.l == 0:
                    ein_out = torch.einsum(f"{z}uv,zui,zv->zvi", w_out, x1_out, x2_out.reshape(batch_out, mul_ir_in2.dim))
                    ein_right = torch.einsum(f"{z}uv,ij,vw,zv->zuiwj", w_right, i1_right, e2_right, x2_right.reshape(batch_right, mul_ir_in2.dim))
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
                    ein_out = torch.einsum("zu,zv->zv", x1_out.reshape(batch_out, mul_ir_in1.dim), x2_out.reshape(batch_out, mul_ir_in2.dim))
                elif specialized_code and mul_ir_in1.ir.l == 0:
                    ein_out = torch.einsum("zu,zvj->zvj", x1_out.reshape(batch_out, mul_ir_in1.dim), x2_out)
                elif specialized_code and mul_ir_in2.ir.l == 0:
                    ein_out = torch.einsum("zui,zv->zvi", x1_out, x2_out.reshape(batch_out, mul_ir_in2.dim))
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
                    ein_out = torch.einsum(f"{z}uw,zu,zu->zw", w_out, x1_out.reshape(batch_out, mul_ir_in1.dim), x2_out.reshape(batch_out, mul_ir_in2.dim))
                elif specialized_code and mul_ir_in1.ir.l == 0:
                    ein_out = torch.einsum(f"{z}uw,zu,zuj->zwj", w_out, x1_out.reshape(batch_out, mul_ir_in1.dim), x2_out)
                elif specialized_code and mul_ir_in2.ir.l == 0:
                    ein_out = torch.einsum(f"{z}uw,zui,zu->zwi", w_out, x1_out, x2_out.reshape(batch_out, mul_ir_in2.dim))
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
                    ein_out = torch.einsum(f"{z}u,zu,zu->zu", w_out, x1_out.reshape(batch_out, mul_ir_in1.dim), x2_out.reshape(batch_out, mul_ir_in2.dim))
                    ein_right = torch.einsum(f"{z}u,uw,zu->zuw", w_right, e2_right, x2_right.reshape(batch_right, mul_ir_in2.dim))
                elif specialized_code and key == (1, 1, 1) and normalization == "component":
                    ein_out = torch.einsum(
                        f"{z}u,zui->zui",
                        w_out,
                        torch.cross(x1_out, x2_out, dim=2)
                    ) / sqrt(2)
                    # For cross product, use the general case right()
                    ein_right = torch.einsum(f"{z}u,ijk,uw,zuj->zuiwk", w_right, w3j_right, e1_right, x2_right)
                elif specialized_code and mul_ir_in1.ir.l == 0:
                    ein_out = torch.einsum(f"{z}u,zu,zuj->zuj", w_out, x1_out.reshape(batch_out, mul_ir_in1.dim), x2_out)
                    ein_right = torch.einsum(f"{z}u,uw,zui->zuwi", w_right, e2_right, x2_right)
                elif specialized_code and mul_ir_in2.ir.l == 0:
                    ein_out = torch.einsum(f"{z}u,zui,zu->zui", w_out, x1_out, x2_out.reshape(batch_out, mul_ir_in2.dim))
                    ein_right = torch.einsum(f"{z}u,ij,uw,zu->zuiwj", w_right, i1_right, e2_right, x2_right.reshape(batch_right, mul_ir_in2.dim))
                elif specialized_code and mul_ir_out.ir.l == 0:
                    ein_out = torch.einsum(f"{z}u,zui,zui->zu", w_out, x1_out, x2_out) / sqrt(mul_ir_in1.ir.dim)**exp
                    ein_right = torch.einsum(f"{z}u,uw,zui->zuiw", w_right, e2_right, x2_right) / sqrt(mul_ir_in1.ir.dim)**exp
                else:
                    ein_out = torch.einsum(f"{z}u,ijk,zuij->zuk", w_out, w3j_out, xx)
                    ein_right = torch.einsum(f"{z}u,ijk,uw,zuj->zuiwk", w_right, w3j_right, e1_right, x2_right)
            else:
                if specialized_code and key == (0, 0, 0):
                    ein_out = torch.einsum("zu,zu->zu", x1_out.reshape(batch_out, mul_ir_in1.dim), x2_out.reshape(batch_out, mul_ir_in2.dim))
                    ein_right = torch.einsum("uw,zu->zuw", e2_right, x2_right.reshape(batch_right, mul_ir_in2.dim))
                elif specialized_code and key == (1, 1, 1) and normalization == "component":
                    ein_out = torch.cross(x1_out, x2_out, dim=2) * (1.0 / sqrt(2))
                    # For cross product, use the general case right()
                    ein_right = torch.einsum("ijk,uw,zuj->zuiwk", w3j_right, e1_right, x2_right)
                elif specialized_code and mul_ir_in1.ir.l == 0:
                    ein_out = torch.einsum("zu,zuj->zuj", x1_out.reshape(batch_out, mul_ir_in1.dim), x2_out)
                    ein_right = torch.einsum("uw,zui->zuwi", e2_right, x2_right)
                elif specialized_code and mul_ir_in2.ir.l == 0:
                    ein_out = torch.einsum("zui,zu->zui", x1_out, x2_out.reshape(batch_out, mul_ir_in2.dim))
                    ein_right = torch.einsum("ij,uw,zu->zuiwj", i1_right, e2_right, x2_right.reshape(batch_right, mul_ir_in2.dim))
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

        ein_out = alpha * ein_out
        ein_right = alpha * ein_right

        out_list_out += [ein_out.reshape(batch_out, mul_ir_out.dim)]
        out_list_right += [ein_right.reshape(batch_right, mul_ir_in1.dim, mul_ir_out.dim)]

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

    # = Return the result =
    out_out = [
        _sum_tensors(
            [out for ins, out in zip(instructions, out_list_out) if ins.i_out == i_out],
            shape=(batch_out, mul_ir_out.dim),
            like=x1s_out
        )
        for i_out, mul_ir_out in enumerate(irreps_out)
        if mul_ir_out.mul > 0
    ]
    if len(out_out) > 1:
        out_out = torch.cat(out_out, dim=1)
    else:
        # Avoid an unnecessary copy in a size one torch.cat
        out_out = out_out[0]

    out_right = [
        torch.cat([
            _sum_tensors(
                [out for ins, out in zip(instructions, out_list_right) if (ins.i_in1, ins.i_out) == (i_in1, i_out)],
                shape=(batch_right, mul_ir_in1.dim, mul_ir_out.dim),
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

    out_out = out_out.reshape(outsize_out)
    out_right = out_right.reshape(outsize_right)

    graph_out.output(out_out.node, torch.Tensor)
    graph_right.output(out_right.node, torch.Tensor)

    # check graphs
    graph_out.lint()
    graph_right.lint()

    # TODO: when eliminate_dead_code() is in PyTorch stable, use that

    if optimize_einsums:
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
        # We use float32 and zeros to save memory and time, since opt_einsum_fx looks only at traced shapes, not values or dtypes.
        batchdim = 4
        example_inputs = (
            torch.zeros((batchdim, irreps_in1.dim), dtype=torch.float32),
            torch.zeros((batchdim, irreps_in2.dim), dtype=torch.float32),
            torch.zeros(
                1 if shared_weights else batchdim,
                flat_weight_index,
                dtype=torch.float32
            ),
            torch.zeros(sum(w3j_dim(*k) for k in w3j), dtype=torch.float32)
        )

        graph_out = jitable(optimize_einsums_full(graph_out, example_inputs))
        graph_right = jitable(optimize_einsums_full(graph_right, example_inputs[1:]))

    return graph_out, graph_right, w3j
