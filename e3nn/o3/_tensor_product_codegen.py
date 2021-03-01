from math import sqrt
from collections import namedtuple
import textwrap

import torch
from e3nn import o3


def _prod(x):
    out = 1
    for a in x:
        out *= a
    return out


def codegen_tensor_product(in1, in2, out, instructions, normalization='component', shared_weights=None, specialized_code=True):
    try:
        in1 = o3.Irreps(in1)
    except AssertionError:
        pass
    try:
        in2 = o3.Irreps(in2)
    except AssertionError:
        pass
    try:
        out = o3.Irreps(out)
    except AssertionError:
        pass

    in1 = [x if len(x) == 3 else x + (1.0,) for x in in1]
    in2 = [x if len(x) == 3 else x + (1.0,) for x in in2]
    out = [x if len(x) == 3 else x + (1.0,) for x in out]

    irreps_in1 = o3.Irreps([(mul, ir) for mul, ir, _var in in1])
    irreps_in2 = o3.Irreps([(mul, ir) for mul, ir, _var in in2])
    irreps_out = o3.Irreps([(mul, ir) for mul, ir, _var in out])

    in1_var = [var for _, _, var in in1]
    in2_var = [var for _, _, var in in2]
    out_var = [var for _, _, var in out]

    # === Build instructions ===
    Instruction = namedtuple("Instruction", "i_in1, i_in2, i_out, connection_mode, has_weight, path_weight, path_shape")
    instructions = [x if len(x) == 6 else x + (1.0,) for x in instructions]
    instructions = [
        Instruction(
            i_in1, i_in2, i_out, connection_mode, has_weight, path_weight,
            {
                'uvw': (irreps_in1[i_in1].mul, irreps_in2[i_in2].mul, irreps_out[i_out].mul),
                'uvu': (irreps_in1[i_in1].mul, irreps_in2[i_in2].mul),
                'uvv': (irreps_in1[i_in1].mul, irreps_in2[i_in2].mul),
                'uuw': (irreps_in1[i_in1].mul, irreps_out[i_out].mul),
                'uuu': (irreps_in1[i_in1].mul,),
                'uvuv': (irreps_in1[i_in1].mul, irreps_in2[i_in2].mul),
            }[connection_mode],
        )
        for i_in1, i_in2, i_out, connection_mode, has_weight, path_weight in instructions
    ]
    weight_numel = sum(_prod(ins.path_shape) for ins in instructions if ins.has_weight)

    # === Init everything for codegen ===
    # == TorchScript main operation templates ==
    code_header = textwrap.dedent("""
    from typing import List
    import torch
    """)
    # The if-else block is needed to avoid an internal TorchScript compiler bug related to the early return.
    funcdef_out = textwrap.dedent(f"""
    @torch.jit.script
    def main(x1: torch.Tensor, x2: torch.Tensor, ws: torch.Tensor, w3j: torch.Tensor) -> torch.Tensor:
        {'x1, x2 = torch.broadcast_tensors(x1[..., :, None], x2[..., None, :])' if shared_weights else ''}
        {'x1, x2, ws = torch.broadcast_tensors(x1[..., :, None, None], x2[..., None, :, None], ws[..., None, None, :])' if not shared_weights else ''}
    """)

    funcdef_right = textwrap.dedent(f"""
    @torch.jit.script
    def main(x2: torch.Tensor, ws: torch.Tensor, w3j: torch.Tensor) -> torch.Tensor:
        {'x2, ws = torch.broadcast_tensors(x2[..., :, None], ws[..., None, :])' if not shared_weights else ''}
    """)

    if irreps_in1.dim == 0 or irreps_in2.dim == 0 or irreps_out.dim == 0:
        funcdef_out += f"    return x1.new_zeros(x1.shape[{':-2' if shared_weights else ':-3'}] + ({irreps_out.dim},))"
        funcdef_right += f"    return x2.new_zeros(x2.shape[{':-1' if shared_weights else ':-2'}] + ({irreps_in1.dim}, {irreps_out.dim},))"
    else:
        funcdef_out += f"""
    {'x1, x2 = x1[..., :, 0], x2[..., 0, :]' if shared_weights else ''}
    {'x1, x2, ws = x1[..., :, 0, 0], x2[..., 0, :, 0], ws[..., 0, 0, :]' if not shared_weights else ''}
    size = x1.shape[:-1]
    outsize = size + ({irreps_out.dim},)
    assert x1.shape[-1] == {irreps_in1.dim}, "Incorrect feature dimension for x1"
    assert x2.shape[-1] == {irreps_in2.dim}, "Incorrect feature dimension for x2"

    x1 = x1.reshape(-1, {irreps_in1.dim})
    x2 = x2.reshape(-1, {irreps_in2.dim})
    ws = ws.reshape(-1, {weight_numel})

    if x1.shape[0] == 0:
        return x1.new_zeros(outsize)
    else:
        batch = x1.shape[0]
        out = x1.new_zeros((batch, {irreps_out.dim}))
        ein = torch.einsum
"""
        funcdef_right += f"""
    {'x2, ws = x2[..., :, 0], ws[..., 0, :]' if not shared_weights else ''}
    size = x2.shape[:-1]
    outsize = size + ({irreps_in1.dim}, {irreps_out.dim},)
    assert x2.shape[-1] == {irreps_in2.dim}, "Incorrect feature dimension for x2"

    x2 = x2.reshape(-1, {irreps_in2.dim})
    ws = ws.reshape(-1, {weight_numel})

    if x2.shape[0] == 0:
        return x2.new_zeros(outsize)
    else:
        batch = x2.shape[0]
        out = x2.new_zeros((batch, {irreps_in1.dim}, {irreps_out.dim}))
        ein = torch.einsum
"""

    # == end TorchScript templates ==
    # Put everything in the else block
    code_out = ""
    code_right = ""
    base_indent = 2
    def indent_for_level(indent_level):
        return ((base_indent + indent_level) * 4) * " "
    s = indent_for_level(0)

    wigners = []

    for i_1, (mul_1, (l_1, p_1)) in enumerate(irreps_in1):
        index_1 = irreps_in1[:i_1].dim
        dim_1 = mul_1 * (2 * l_1 + 1)
        code_out += f"{s}x1_{i_1} = x1[:, {index_1}:{index_1+dim_1}].reshape(batch, {mul_1}, {2 * l_1 + 1})\n"
    code_out += "\n"

    for i_2, (mul_2, (l_2, p_2)) in enumerate(irreps_in2):
        index_2 = irreps_in2[:i_2].dim
        dim_2 = mul_2 * (2 * l_2 + 1)
        line = f"{s}x2_{i_2} = x2[:, {index_2}:{index_2+dim_2}].reshape(batch, {mul_2}, {2 * l_2 + 1})\n"
        code_out += line
        code_right += line
    code_out += "\n"
    code_right += "\n"

    z = '' if shared_weights else 'z'
    last_ss = None

    index_w = -1
    flat_weight_i = 0

    for ins in instructions:
        mul_1, (l_1, p_1) = irreps_in1[ins.i_in1]
        mul_2, (l_2, p_2) = irreps_in2[ins.i_in2]
        mul_out, (l_out, p_out) = irreps_out[ins.i_out]
        dim_1 = mul_1 * (2 * l_1 + 1)
        dim_2 = mul_2 * (2 * l_2 + 1)
        dim_out = mul_out * (2 * l_out + 1)
        index_1 = irreps_in1[:ins.i_in1].dim
        index_2 = irreps_in2[:ins.i_in2].dim
        index_out = irreps_out[:ins.i_out].dim

        assert p_1 * p_2 == p_out
        assert abs(l_1 - l_2) <= l_out <= l_1 + l_2

        if dim_1 == 0 or dim_2 == 0 or dim_out == 0:
            continue

        alpha = ins.path_weight * out_var[ins.i_out] / sum(in1_var[i.i_in1] * in2_var[i.i_in2] for i in instructions if i.i_out == ins.i_out)

        s = indent_for_level(0)

        line = (
            f"{s}with torch.autograd.profiler.record_function("
            f"'{irreps_in1[ins.i_in1:ins.i_in1+1]} x {irreps_in2[ins.i_in2:ins.i_in2+1]} "
            f"= {irreps_out[ins.i_out:ins.i_out+1]} {ins.connection_mode} {ins.has_weight}'):\n"
        )
        code_out += line
        code_right += line

        s = indent_for_level(1)

        code_out += f"{s}s1 = x1_{ins.i_in1}\n"
        code_right += f"{s}e1 = torch.eye({mul_1}, dtype=x2.dtype, device=x2.device)\n"

        line = f"{s}s2 = x2_{ins.i_in2}\n"
        code_out += line
        code_right += line

        assert ins.connection_mode in ['uvw', 'uvu', 'uvv', 'uuw', 'uuu', 'uvuv']

        alpha = sqrt(alpha / {
            'uvw': (mul_1 * mul_2),
            'uvu': mul_2,
            'uvv': mul_1,
            'uuw': mul_1,
            'uuu': 1,
            'uvuv': 1,
        }[ins.connection_mode])

        if ins.has_weight:
            index_w += 1
            # Extract the weight from the flattened weight tensor
            line = f"{s}ws_{index_w} = ws[:, {flat_weight_i}:{flat_weight_i + _prod(ins.path_shape)}].reshape({(() if shared_weights else (-1,)) + tuple(ins.path_shape)})\n"
            code_out += line
            code_right += line
            flat_weight_i += _prod(ins.path_shape)

        line_out = f"{s}out[:, {index_out}:{index_out+dim_out}] += {alpha} * {{}}.reshape(batch, {dim_out})\n\n"
        line_right = f"{s}out[:, {index_1}:{index_1+dim_1}, {index_out}:{index_out+dim_out}] += {alpha} * {{}}.reshape(batch, {dim_1}, {dim_out})\n\n"

        if specialized_code:
            # optimized code for special cases:
            # 0 x 0 = 0
            # 0 x L = L
            # L x 0 = L
            # L x L = 0
            # 1 x 1 = 1

            if (l_1, l_2, l_out) == (0, 0, 0) and ins.connection_mode in ['uvw', 'uvu'] and normalization in ['component', 'norm'] and ins.has_weight:
                code_out += f"{s}s1 = s1.reshape(batch, {mul_1})\n"
                line = f"{s}s2 = s2.reshape(batch, {mul_2})\n"
                code_out += line
                code_right += line

                if ins.connection_mode == 'uvw':
                    code_out += line_out.format(f"ein('{z}uvw,zu,zv->zw', ws_{index_w}, s1, s2)")
                    code_right += line_right.format(f"ein('{z}uvw,zv->zuw', ws_{index_w}, s2)")
                if ins.connection_mode == 'uvu':
                    code_out += line_out.format(f"ein('{z}uv,zu,zv->zu', ws_{index_w}, s1, s2)")
                    code_right += line_right.format(f"ein('{z}uv,uw,zv->zuw', ws_{index_w}, e1, s2)")

                continue

            if l_1 == 0 and l_2 == l_out and ins.connection_mode in ['uvw', 'uvu'] and normalization == 'component' and ins.has_weight:
                code_out += f"{s}s1 = s1.reshape(batch, {mul_1})\n"

                if ins.connection_mode == 'uvw':
                    code_out += line_out.format(f"ein('{z}uvw,zu,zvi->zwi', ws_{index_w}, s1, s2)")
                    code_right += line_right.format(f"ein('{z}uvw,zvi->zuwi', ws_{index_w}, s2)")
                if ins.connection_mode == 'uvu':
                    code_out += line_out.format(f"ein('{z}uv,zu,zvi->zui', ws_{index_w}, s1, s2)")
                    code_right += line_right.format(f"ein('{z}uv,uw,zvi->zuwi', ws_{index_w}, e1, s2)")

                continue

            if l_1 == l_out and l_2 == 0 and ins.connection_mode in ['uvw', 'uvu'] and normalization == 'component' and ins.has_weight:
                code_out += f"{s}s2 = s2.reshape(batch, {mul_2})\n"
                code_right += f"{s}s2 = s2.reshape(batch, {mul_2})\n"
                code_right += f"{s}wig = torch.eye({2 * l_1 + 1}, dtype=x2.dtype, device=x2.device)\n"

                if ins.connection_mode == 'uvw':
                    code_out += line_out.format(f"ein('{z}uvw,zui,zv->zwi', ws_{index_w}, s1, s2)")
                    code_right += line_right.format(f"ein('{z}uvw,ij,zv->zuiwj', ws_{index_w}, wig, s2)")
                if ins.connection_mode == 'uvu':
                    code_out += line_out.format(f"ein('{z}uv,zui,zv->zui', ws_{index_w}, s1, s2)")
                    code_right += line_right.format(f"ein('{z}uv,ij,uw,zv->zuiwj', ws_{index_w}, wig, e1, s2)")

                continue

            if l_1 == l_2 and l_out == 0 and ins.connection_mode == 'uvw' and normalization == 'component' and ins.has_weight:
                # Cl_l_0 = eye / sqrt(2L+1)
                code_out += line_out.format(f"ein('{z}uvw,zui,zvi->zw', ws_{index_w} / {sqrt(2 * l_1 + 1)}, s1, s2)")
                code_right += line_right.format(f"ein('{z}uvw,zvi->zuiw', ws_{index_w} / {sqrt(2 * l_1 + 1)}, s2)")
                continue

            if l_1 == l_2 and l_out == 0 and ins.connection_mode == 'uvu' and normalization == 'component' and ins.has_weight:
                # Cl_l_0 = eye / sqrt(2L+1)
                code_out += line_out.format(f"ein('{z}uv,zui,zvi->zu', ws_{index_w} / {sqrt(2 * l_1 + 1)}, s1, s2)")
                code_right += line_right.format(f"ein('{z}uv,uw,zvi->zuiw', ws_{index_w} / {sqrt(2 * l_1 + 1)}, e1, s2)")
                continue

            if l_1 == l_2 and l_out == 0 and ins.connection_mode == 'uuu' and normalization == 'component' and ins.has_weight:
                # Cl_l_0 = eye / sqrt(2L+1)
                code_out += line_out.format(f"ein('{z}u,zui,zui->zu', ws_{index_w} / {sqrt(2 * l_1 + 1)}, s1, s2)")
                code_right += line_right.format(f"ein('{z}u,uw,zui->zuiw', ws_{index_w} / {sqrt(2 * l_1 + 1)}, e1, s2)")
                continue

            if l_1 == l_2 and l_out == 0 and ins.connection_mode == 'uuu' and normalization == 'component' and not ins.has_weight:
                # Cl_l_0 = eye / sqrt(2L+1)
                code_out += line_out.format(f"ein('zui,zui->zu', s1, s2).div({sqrt(2 * l_1 + 1)})")
                code_right += line_right.format(f"ein('uw,zui->zuiw', e1, s2).div({sqrt(2 * l_1 + 1)})")
                continue

            if (l_1, l_2, l_out) == (1, 1, 1) and ins.connection_mode == 'uvw' and normalization == 'component' and ins.has_weight:
                # C1_1_1 = levi-civita / sqrt(2)
                code_out += f"{s}s1 = s1.reshape(batch, {mul_1}, 1, {2 * l_1 + 1})\n"
                code_out += f"{s}s2 = s2.reshape(batch, 1, {mul_2}, {2 * l_2 + 1})\n"
                code_out += f"{s}s1, s2 = torch.broadcast_tensors(s1, s2)\n"
                code_out += line_out.format(f"ein('{z}uvw,zuvi->zwi', ws_{index_w} / {sqrt(2)}, torch.cross(s1, s2, dim=3))")

                if (l_1, l_2, l_out) in wigners:
                    index_w3j = wigners.index((l_1, l_2, l_out))
                else:
                    index_w3j = len(wigners)
                    wigners += [(l_1, l_2, l_out)]

                code_right += line_right.format(f"ein('{z}uvw,ijk,zvj->zuiwk', ws_{index_w}, w3j_{index_w3j}, s2)")
                continue

            if (l_1, l_2, l_out) == (1, 1, 1) and ins.connection_mode == 'uvu' and normalization == 'component' and ins.has_weight:
                # C1_1_1 = levi-civita / sqrt(2)
                code_out += f"{s}s1 = s1.reshape(batch, {mul_1}, 1, {2 * l_1 + 1})\n"
                code_out += f"{s}s2 = s2.reshape(batch, 1, {mul_2}, {2 * l_2 + 1})\n"
                code_out += f"{s}s1, s2 = torch.broadcast_tensors(s1, s2)\n"
                code_out += line_out.format(f"ein('{z}uv,zuvi->zui', ws_{index_w} / {sqrt(2)}, torch.cross(s1, s2, dim=3))")

                if (l_1, l_2, l_out) in wigners:
                    index_w3j = wigners.index((l_1, l_2, l_out))
                else:
                    index_w3j = len(wigners)
                    wigners += [(l_1, l_2, l_out)]

                code_right += line_right.format(f"ein('{z}uv,ijk,uw,zvj->zuiwk', ws_{index_w}, w3j_{index_w3j}, e1, s2)")
                continue
        # == end specialized code ==

        if last_ss != (ins.i_in1, ins.i_in2, ins.connection_mode[:2]):
            if ins.connection_mode[:2] == 'uv':
                code_out += f"{s}ss = ein('zui,zvj->zuvij', s1, s2)\n"
            if ins.connection_mode[:2] == 'uu':
                code_out += f"{s}ss = ein('zui,zuj->zuij', s1, s2)\n"
            last_ss = (ins.i_in1, ins.i_in2, ins.connection_mode[:2])

        if (l_1, l_2, l_out) in wigners:
            index_w3j = wigners.index((l_1, l_2, l_out))
        else:
            index_w3j = len(wigners)
            wigners += [(l_1, l_2, l_out)]

        if ins.connection_mode == 'uvw':
            assert ins.has_weight
            code_out += line_out.format(f"ein('{z}uvw,ijk,zuvij->zwk', ws_{index_w}, w3j_{index_w3j}, ss)")
            code_right += line_right.format(f"ein('{z}uvw,ijk,zvj->zuiwk', ws_{index_w}, w3j_{index_w3j}, s2)")
        if ins.connection_mode == 'uvu':
            assert mul_1 == mul_out
            if ins.has_weight:
                code_out += line_out.format(f"ein('{z}uv,ijk,zuvij->zuk', ws_{index_w}, w3j_{index_w3j}, ss)")
                code_right += line_right.format(f"ein('{z}uv,ijk,uw,zvj->zuiwk', ws_{index_w}, w3j_{index_w3j}, e1, s2)")
            else:
                code_out += line_out.format(f"ein('ijk,zuvij->zuk', w3j_{index_w3j}, ss)")
                code_right += line_right.format(f"ein('ijk,uw,zvj->zuiwk', w3j_{index_w3j}, e1, s2)")
        if ins.connection_mode == 'uvv':
            assert mul_2 == mul_out
            if ins.has_weight:
                code_out += line_out.format(f"ein('{z}uv,ijk,zuvij->zvk', ws_{index_w}, w3j_{index_w3j}, ss)")
                code_right += line_right.format(f"ein('{z}uv,ijk,zvj->zuivk', ws_{index_w}, w3j_{index_w3j}, s2)")
            else:
                code_out += line_out.format(f"ein('ijk,zuvij->zvk', w3j_{index_w3j}, ss)")
                code_right += line_right.format(f"ein('u,ijk,zvj->zuivk', s2.new_zeros({mul_1}).fill_(1.0), w3j_{index_w3j}, s2)")
        if ins.connection_mode == 'uuw':
            assert mul_1 == mul_2
            if ins.has_weight:
                code_out += line_out.format(f"ein('{z}uw,ijk,zuij->zwk', ws_{index_w}, w3j_{index_w3j}, ss)")
                code_right += line_right.format(f"ein('{z}uw,ijk,zuj->zuiwk', ws_{index_w}, w3j_{index_w3j}, s2)")
            else:
                assert mul_out == 1
                code_out += line_out.format(f"ein('ijk,zuij->zk', w3j_{index_w3j}, ss)")
                code_right += line_right.format(f"ein('ijk,zuj->zuik', w3j_{index_w3j}, s2)")
        if ins.connection_mode == 'uuu':
            assert mul_1 == mul_2 == mul_out
            if ins.has_weight:
                code_out += line_out.format(f"ein('{z}u,ijk,zuij->zuk', ws_{index_w}, w3j_{index_w3j}, ss)")
                code_right += line_right.format(f"ein('{z}u,ijk,uw,zuj->zuiwk', ws_{index_w}, w3j_{index_w3j}, e1, s2)")
            else:
                code_out += line_out.format(f"ein('ijk,zuij->zuk', w3j_{index_w3j}, ss)")
                code_right += line_right.format(f"ein('ijk,uw,zuj->zuiwk', w3j_{index_w3j}, e1, s2)")
        if ins.connection_mode == 'uvuv':
            assert mul_1 * mul_2 == mul_out
            if ins.has_weight:
                code_out += line_out.format(f"ein('{z}uv,ijk,zuvij->zuvk', ws_{index_w}, w3j_{index_w3j}, ss)")
                code_right += line_right.format(f"ein('{z}uv,ijk,uw,zvj->zuiwvk', ws_{index_w}, w3j_{index_w3j}, e1, s2)")
            else:
                code_out += line_out.format(f"ein('ijk,zuvij->zuvk', w3j_{index_w3j}, ss)")
                code_right += line_right.format(f"ein('ijk,uw,zvj->zuiwvk', w3j_{index_w3j}, e1, s2)")
        code_out += "\n"

    code_out += f"{s}return out.reshape(outsize)"
    code_right += f"{s}return out.reshape(outsize)"

    flat_wigner_index = 0
    code_wigners = []
    s = indent_for_level(0)
    for i, (l_1, l_2, l_out) in enumerate(wigners):
        shape = (2 * l_1 + 1, 2 * l_2 + 1, 2 * l_out + 1)
        code_wigners.append(f"{s}w3j_{i} = w3j[{flat_wigner_index}:{flat_wigner_index + _prod(shape)}].reshape({tuple(shape)})")
        flat_wigner_index += _prod(shape)
    code_wigners = '\n'.join(code_wigners)

    # Finalize the code
    if irreps_in1.dim == 0 or irreps_in2.dim == 0 or irreps_out.dim == 0:
        full_code_out = "\n\n".join([code_header, funcdef_out])
        full_code_right = "\n\n".join([code_header, funcdef_right])
    else:
        full_code_out = "\n\n".join([code_header, funcdef_out, code_wigners, code_out])
        full_code_right = "\n\n".join([code_header, funcdef_right, code_wigners, code_right])

    return irreps_in1, irreps_in2, irreps_out, full_code_out, full_code_right, instructions, wigners
