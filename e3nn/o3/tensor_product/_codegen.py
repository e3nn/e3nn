from math import sqrt
from typing import List, Tuple

import torch
from torch import fx

from e3nn import o3
from e3nn.util import prod

from ._instruction import Instruction


def _get_code(graph):
    x = graph.python_code('')
    x = x.replace('def forward(self, ', '@torch.jit.script\ndef main(')
    x = x.replace('Ellipsis', '...')
    return x


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
    codegen_kwargs: dict = {}
) -> Tuple[str, str, list]:
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

    if shared_weights:
        x1s_out, x2s_out = torch.broadcast_tensors(x1s_out[..., :, None], x2s_out[..., None, :])
    else:
        x1s_out, x2s_out, ws_out = torch.broadcast_tensors(x1s_out[..., :, None, None], x2s_out[..., None, :, None], ws_out[..., None, None, :])
        x2s_right, ws_right = torch.broadcast_tensors(x2s_right[..., :, None], ws_right[..., None, :])

    # = Short-circut for zero dimensional =
    if irreps_in1.dim == 0 or irreps_in2.dim == 0 or irreps_out.dim == 0:
        out_out = x1s_out.new_zeros(x1s_out.shape[:-2 if shared_weights else -3] + (irreps_out.dim,))
        out_right = x2s_right.new_zeros(x2s_right.shape[:-1 if shared_weights else -2] + (irreps_in1.dim, irreps_out.dim,))

        graph_out.output(out_out.node, torch.Tensor)
        graph_right.output(out_right.node, torch.Tensor)
        # Short circut
        # the empty list is wigners
        return _get_code(graph_out), _get_code(graph_right), []

    # = Broadcast inputs =
    if shared_weights:
        x1s_out, x2s_out = x1s_out[..., :, 0], x2s_out[..., 0, :]
    else:
        x1s_out, x2s_out, ws_out = x1s_out[..., :, 0, 0], x2s_out[..., 0, :, 0], ws_out[..., 0, 0, :]
        x2s_right, ws_right = x2s_right[..., :, 0], ws_right[..., 0, :]

    outsize_out = x1s_out.shape[:-1] + (irreps_out.dim,)
    outsize_right = x2s_right.shape[:-1] + (irreps_in1.dim, irreps_out.dim,)

    # assert x1s_out.shape[-1] == irreps_in1.dim, "Incorrect feature dimension for x1"
    # assert x2s_out.shape[-1] == irreps_in2.dim, "Incorrect feature dimension for x2"
    # assert x2s_right.shape[-1] == {irreps_in2.dim}, "Incorrect feature dimension for x2"

    x1s_out = x1s_out.reshape(-1, irreps_in1.dim)
    x2s_out = x2s_out.reshape(-1, irreps_in2.dim)
    x2s_right = x2s_right.reshape(-1, irreps_in2.dim)

    batch_out = x1s_out.shape[0]
    batch_right = x2s_right.shape[0]

    weight_numel = sum(prod(ins.path_shape) for ins in instructions if ins.has_weight)
    if weight_numel > 0:
        ws_out = ws_out.reshape(-1, weight_numel)
        ws_right = ws_right.reshape(-1, weight_numel)
    del weight_numel

    # = extract wigners =
    w3j = []
    w3j_dict_out = dict()
    w3j_dict_right = dict()

    def w3j_dim(l1, l2, l3):
        return (2 * l1 + 1) * (2 * l2 + 1) * (2 * l3 + 1)

    # = extract individual input irreps =
    x1_list_out = [
        x1s_out[:, i].reshape(batch_out, mul_ir.mul, mul_ir.ir.dim)
        for i, mul_ir in zip(irreps_in1.slices(), irreps_in1)
    ]

    x2_list_out = []
    x2_list_right = []
    for i, mul_ir in zip(irreps_in2.slices(), irreps_in2):
        x2_list_out.append(x2s_out[:, i].reshape(batch_out, mul_ir.mul, mul_ir.ir.dim))
        x2_list_right.append(x2s_right[:, i].reshape(batch_right, mul_ir.mul, mul_ir.ir.dim))

    z = '' if shared_weights else 'z'
    xx_dict = dict()

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

        # We didn't make this instruction specialized, so do the general case
        key = (ins.i_in1, ins.i_in2, ins.connection_mode[:2])
        if key not in xx_dict:
            if ins.connection_mode[:2] == 'uv':
                xx_dict[key] = torch.einsum('zui,zvj->zuvij', x1_out, x2_out)
            if ins.connection_mode[:2] == 'uu':
                xx_dict[key] = torch.einsum('zui,zuj->zuij', x1_out, x2_out)
        xx = xx_dict[key]

        key = (mul_ir_in1.ir.l, mul_ir_in2.ir.l, mul_ir_out.ir.l)
        if key not in w3j:
            i = sum(w3j_dim(*k) for k in w3j)
            w3j_dict_out[key] = w3js_out[i:i + w3j_dim(*key)].reshape(mul_ir_in1.ir.dim, mul_ir_in2.ir.dim, mul_ir_out.ir.dim)
            w3j_dict_right[key] = w3js_right[i:i + w3j_dim(*key)].reshape(mul_ir_in1.ir.dim, mul_ir_in2.ir.dim, mul_ir_out.ir.dim)
            w3j.append(key)
        w3j_out = w3j_dict_out[key]
        w3j_right = w3j_dict_right[key]

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
                exp = {'component': 1, 'norm': -1}[normalization]
                ein_out = torch.einsum(f"{z}uvw,zui,zvi->zw", w_out, x1_out, x2_out) / sqrt(mul_ir_in1.ir.dim)**exp
                ein_right = torch.einsum(f"{z}uvw,zvi->zuiw", w_right, x2_right) / sqrt(mul_ir_in1.ir.dim)**exp
            else:
                ein_out = torch.einsum(f"{z}uvw,ijk,zuvij->zwk", w_out, w3j_out, xx)
                ein_right = torch.einsum(f"{z}uvw,ijk,zvj->zuiwk", w_right, w3j_right, x2_right)
        if ins.connection_mode == 'uvu':
            assert mul_ir_in1.mul == mul_ir_out.mul
            if ins.has_weight:
                # TODO implement specialized code
                ein_out = torch.einsum(f"{z}uv,ijk,zuvij->zuk", w_out, w3j_out, xx)
                ein_right = torch.einsum(f"{z}uv,ijk,uw,zvj->zuiwk", w_right, w3j_right, e1_right, x2_right)
            else:
                # TODO implement specialized code
                ein_out = torch.einsum("ijk,zuvij->zuk", w3j_out, xx)
                ein_right = torch.einsum("ijk,uw,zvj->zuiwk", w3j_right, e1_right, x2_right)
        if ins.connection_mode == 'uvv':
            assert mul_ir_in2.mul == mul_ir_out.mul
            if ins.has_weight:
                # TODO implement specialized code
                ein_out = torch.einsum(f"{z}uv,ijk,zuvij->zvk", w_out, w3j_out, xx)
                ein_right = torch.einsum(f"{z}uv,ijk,zvj->zuivk", w_right, w3j_right, x2_right)
            else:
                # TODO implement specialized code
                ein_out = torch.einsum("ijk,zuvij->zvk", w3j_out, xx)
                s2ones = fx.Proxy(graph_right.call_function(torch.ones, (mul_ir_in1.mul,), dict(device=x2_right.device.node, dtype=x2_right.dtype.node)))
                ein_right = torch.einsum("u,ijk,zvj->zuivk", s2ones, w3j_right, x2_right)
        if ins.connection_mode == 'uuw':
            assert mul_ir_in1.mul == mul_ir_in2.mul
            if ins.has_weight:
                # TODO implement specialized code
                ein_out = torch.einsum(f"{z}uw,ijk,zuij->zwk", w_out, w3j_out, xx)
                ein_right = torch.einsum(f"{z}uw,ijk,zuj->zuiwk", w_right, w3j_right, x2_right)
            else:
                # TODO implement specialized code
                assert mul_ir_out.mul == 1
                ein_out = torch.einsum("ijk,zuij->zk", w3j_out, xx)
                ein_right = torch.einsum("ijk,zuj->zuik", w3j_right, x2_right)
        if ins.connection_mode == 'uuu':
            assert mul_ir_in1.mul == mul_ir_in2.mul == mul_ir_out.mul
            if ins.has_weight:
                # TODO implement specialized code
                ein_out = torch.einsum(f"{z}u,ijk,zuij->zuk", w_out, w3j_out, xx)
                ein_right = torch.einsum(f"{z}u,ijk,uw,zuj->zuiwk", w_right, w3j_right, e1_right, x2_right)
            else:
                # TODO implement specialized code
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

    # = Return the result =
    out_out = torch.cat([
        _sum_tensors(
            [out for ins, out in zip(instructions, out_list_out) if ins.i_out == i_out],
            shape=(batch_out, mul_ir_out.dim),
            like=x1s_out
        )
        for i_out, mul_ir_out in enumerate(irreps_out)
    ], dim=1)

    out_right = torch.cat([
        torch.cat([
            _sum_tensors(
                [out for ins, out in zip(instructions, out_list_right) if (ins.i_in1, ins.i_out) == (i_in1, i_out)],
                shape=(batch_right, mul_ir_in1.dim, mul_ir_out.dim),
                like=x2s_right
            )
            for i_out, mul_ir_out in enumerate(irreps_out)
        ], dim=2)
        for i_in1, mul_ir_in1 in enumerate(irreps_in1)
    ], dim=1)

    out_out = out_out.reshape(outsize_out)
    out_right = out_right.reshape(outsize_right)

    graph_out.output(out_out.node, torch.Tensor)
    graph_right.output(out_right.node, torch.Tensor)

    try:
        from opt_einsum_fx import optimize_einsums, jitable

        example_inputs = (
            irreps_in1.randn(4, -1, dtype=torch.float32),
            irreps_in2.randn(4, -1, dtype=torch.float32),
            torch.randn(1 if shared_weights else 4, flat_weight_index, dtype=torch.float32),
            torch.randn(sum(w3j_dim(*k) for k in w3j), dtype=torch.float32)
        )

        m = fx.GraphModule(torch.nn.Module(), graph_out)
        m = jitable(optimize_einsums(m, example_inputs))
        graph_out = m.graph

        m = fx.GraphModule(torch.nn.Module(), graph_right)
        m = jitable(optimize_einsums(m, example_inputs[1:]))
        graph_right = m.graph
    except:
        pass

    return _get_code(graph_out), _get_code(graph_right), w3j
