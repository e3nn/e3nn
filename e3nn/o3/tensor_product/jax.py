from math import sqrt
from typing import List

from e3nn import o3
from e3nn.util import prod

from ._instruction import Instruction

import opt_einsum as oe
import jax.numpy as jnp


def _sum_tensors(xs, shape):
    if len(xs) > 0:
        out = xs[0]
        for x in xs[1:]:
            out = out + x
        return out
    return jnp.zeros(shape)


def tensor_product(
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
):
    einsum = oe.contract if optimize_einsums else jnp.einsum

    def f(x1s, x2s, ws, w3js):
        # TODO add broadcast
        if shared_weights:
            # size = jnp.broadcast_shapes(x1s.shape[:-1], x2s.shape[:-1])
            size = x1s.shape[:-1]
        else:
            # size = jnp.broadcast_shapes(x1s.shape[:-1], x2s.shape[:-1], ws.shape[:-1])
            size = x1s.shape[:-1]

        # = Short-circut for zero dimensional =
        if irreps_in1.dim == 0 or irreps_in2.dim == 0 or irreps_out.dim == 0:
            return jnp.zeros(size + (irreps_out.dim,))

        # = Broadcast inputs =
        x1s, x2s = jnp.broadcast_to(x1s, size + (irreps_in1.dim,)), jnp.broadcast_to(x2s, size + (irreps_in2.dim,))
        if not shared_weights:
            ws = jnp.broadcast_to(ws, size + (ws.shape[-1],))

        outsize = size + (irreps_out.dim,)

        x1s = x1s.reshape(-1, irreps_in1.dim)
        x2s = x2s.reshape(-1, irreps_in2.dim)

        batch = x1s.shape[0]

        weight_numel = sum(prod(ins.path_shape) for ins in instructions if ins.has_weight)
        if weight_numel > 0:
            ws = ws.reshape(-1, weight_numel)
        del weight_numel

        # = extract wigners =
        w3j_keys = []
        w3j_dict = dict()

        def w3j_dim(l1, l2, l3):
            return (2 * l1 + 1) * (2 * l2 + 1) * (2 * l3 + 1)

        # = extract individual input irreps =
        x1_list = [
            x1s[:, i].reshape(batch, mul_ir.mul, mul_ir.ir.dim)
            for i, mul_ir in zip(irreps_in1.slices(), irreps_in1)
        ]

        x2_list = []
        for i, mul_ir in zip(irreps_in2.slices(), irreps_in2):
            x2_list.append(x2s[:, i].reshape(batch, mul_ir.mul, mul_ir.ir.dim))

        z = '' if shared_weights else 'z'
        xx_dict = dict()

        flat_weight_index = 0

        out_list = []

        for ins in instructions:
            mul_ir_in1 = irreps_in1[ins.i_in1]
            mul_ir_in2 = irreps_in2[ins.i_in2]
            mul_ir_out = irreps_out[ins.i_out]

            assert mul_ir_in1.ir.p * mul_ir_in2.ir.p == mul_ir_out.ir.p
            assert abs(mul_ir_in1.ir.l - mul_ir_in2.ir.l) <= mul_ir_out.ir.l <= mul_ir_in1.ir.l + mul_ir_in2.ir.l

            if mul_ir_in1.dim == 0 or mul_ir_in2.dim == 0 or mul_ir_out.dim == 0:
                continue

            alpha = ins.path_weight * out_var[ins.i_out] / sum(in1_var[i.i_in1] * in2_var[i.i_in2] for i in instructions if i.i_out == ins.i_out)

            x1 = x1_list[ins.i_in1]
            x2 = x2_list[ins.i_in2]

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
                w = ws[:, flat_weight_index:flat_weight_index + prod(ins.path_shape)].reshape((() if shared_weights else (-1,)) + tuple(ins.path_shape))
                flat_weight_index += prod(ins.path_shape)

            # We didn't make this instruction specialized, so do the general case
            key = (ins.i_in1, ins.i_in2, ins.connection_mode[:2])
            if key not in xx_dict:
                if ins.connection_mode[:2] == 'uv':
                    xx_dict[key] = einsum('zui,zvj->zuvij', x1, x2)
                if ins.connection_mode[:2] == 'uu':
                    xx_dict[key] = einsum('zui,zuj->zuij', x1, x2)
            xx = xx_dict[key]

            key = (mul_ir_in1.ir.l, mul_ir_in2.ir.l, mul_ir_out.ir.l)
            if key not in w3j_keys:
                i = sum(w3j_dim(*k) for k in w3j_keys)
                w3j_dict[key] = w3js[i:i + w3j_dim(*key)].reshape(mul_ir_in1.ir.dim, mul_ir_in2.ir.dim, mul_ir_out.ir.dim)
                w3j_keys.append(key)
            w3j = w3j_dict[key]

            exp = {'component': 1, 'norm': -1}[normalization]

            if ins.connection_mode == 'uvw':
                assert ins.has_weight
                if specialized_code and key == (0, 0, 0):
                    ein_out = einsum(f"{z}uvw,zu,zv->zw", w, x1.reshape(batch, mul_ir_in1.dim), x2.reshape(batch, mul_ir_in2.dim))
                elif specialized_code and mul_ir_in1.ir.l == 0:
                    ein_out = einsum(f"{z}uvw,zu,zvj->zwj", w, x1.reshape(batch, mul_ir_in1.dim), x2)
                elif specialized_code and mul_ir_in2.ir.l == 0:
                    ein_out = einsum(f"{z}uvw,zui,zv->zwi", w, x1, x2.reshape(batch, mul_ir_in2.dim))
                elif specialized_code and mul_ir_out.ir.l == 0:
                    ein_out = einsum(f"{z}uvw,zui,zvi->zw", w, x1, x2) / sqrt(mul_ir_in1.ir.dim)**exp
                else:
                    ein_out = einsum(f"{z}uvw,ijk,zuvij->zwk", w, w3j, xx)
            if ins.connection_mode == 'uvu':
                assert mul_ir_in1.mul == mul_ir_out.mul
                if ins.has_weight:
                    if specialized_code and key == (0, 0, 0):
                        ein_out = einsum(f"{z}uv,zu,zv->zu", w, x1.reshape(batch, mul_ir_in1.dim), x2.reshape(batch, mul_ir_in2.dim))
                    elif specialized_code and mul_ir_in1.ir.l == 0:
                        ein_out = einsum(f"{z}uv,zu,zvj->zuj", w, x1.reshape(batch, mul_ir_in1.dim), x2)
                    elif specialized_code and mul_ir_in2.ir.l == 0:
                        ein_out = einsum(f"{z}uv,zui,zv->zui", w, x1, x2.reshape(batch, mul_ir_in2.dim))
                    elif specialized_code and mul_ir_out.ir.l == 0:
                        exp = {'component': 1, 'norm': -1}[normalization]
                        ein_out = einsum(f"{z}uv,zui,zvi->zu", w, x1, x2) / sqrt(mul_ir_in1.ir.dim)**exp
                    else:
                        ein_out = einsum(f"{z}uv,ijk,zuvij->zuk", w, w3j, xx)
                else:
                    # not so useful operation because v is summed
                    ein_out = einsum("ijk,zuvij->zuk", w3j, xx)
            if ins.connection_mode == 'uvv':
                assert mul_ir_in2.mul == mul_ir_out.mul
                if ins.has_weight:
                    if specialized_code and key == (0, 0, 0):
                        ein_out = einsum(f"{z}uv,zu,zv->zv", w, x1.reshape(batch, mul_ir_in1.dim), x2.reshape(batch, mul_ir_in2.dim))
                    elif specialized_code and mul_ir_in1.ir.l == 0:
                        ein_out = einsum(f"{z}uv,zu,zvj->zvj", w, x1.reshape(batch, mul_ir_in1.dim), x2)
                    elif specialized_code and mul_ir_in2.ir.l == 0:
                        ein_out = einsum(f"{z}uv,zui,zv->zvi", w, x1, x2.reshape(batch, mul_ir_in2.dim))
                    elif specialized_code and mul_ir_out.ir.l == 0:
                        ein_out = einsum(f"{z}uv,zui,zvi->zv", w, x1, x2) / sqrt(mul_ir_in1.ir.dim)**exp
                    else:
                        ein_out = einsum(f"{z}uv,ijk,zuvij->zvk", w, w3j, xx)
                else:
                    # not so useful operation because u is summed
                    ein_out = einsum("ijk,zuvij->zvk", w3j, xx)
            if ins.connection_mode == 'uuw':
                assert mul_ir_in1.mul == mul_ir_in2.mul
                if ins.has_weight:
                    # TODO implement specialized code
                    ein_out = einsum(f"{z}uw,ijk,zuij->zwk", w, w3j, xx)
                else:
                    # equivalent to tp(x, y, 'uuu').sum('u')
                    assert mul_ir_out.mul == 1
                    ein_out = einsum("ijk,zuij->zk", w3j, xx)
            if ins.connection_mode == 'uuu':
                assert mul_ir_in1.mul == mul_ir_in2.mul == mul_ir_out.mul
                if ins.has_weight:
                    # TODO implement specialized code
                    ein_out = einsum(f"{z}u,ijk,zuij->zuk", w, w3j, xx)
                else:
                    # TODO implement specialized code
                    ein_out = einsum("ijk,zuij->zuk", w3j, xx)
            if ins.connection_mode == 'uvuv':
                assert mul_ir_in1.mul * mul_ir_in2.mul == mul_ir_out.mul
                if ins.has_weight:
                    # TODO implement specialized code
                    ein_out = einsum(f"{z}uv,ijk,zuvij->zuvk", w, w3j, xx)
                else:
                    # TODO implement specialized code
                    ein_out = einsum("ijk,zuvij->zuvk", w3j, xx)

            ein_out = alpha * ein_out

            out_list += [ein_out.reshape(batch, mul_ir_out.dim)]

        # = Return the result =
        out_out = jnp.concatenate([
            _sum_tensors(
                [out for ins, out in zip(instructions, out_list) if ins.i_out == i_out],
                shape=(batch, mul_ir_out.dim),
            )
            for i_out, mul_ir_out in enumerate(irreps_out)
        ], axis=1)

        return out_out.reshape(outsize)
    return f
