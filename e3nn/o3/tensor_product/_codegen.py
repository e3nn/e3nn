from typing import List, Tuple
from math import sqrt
import textwrap

from e3nn import o3
from e3nn.util.codegen import LazyCodeGenerator
from e3nn.util import prod

from ._instruction import Instruction


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
) -> Tuple[LazyCodeGenerator, LazyCodeGenerator, list]:
    cg_out = LazyCodeGenerator(**codegen_kwargs)
    cg_right = LazyCodeGenerator(**codegen_kwargs)

    weight_numel = sum(prod(ins.path_shape) for ins in instructions if ins.has_weight)

    # = Function definitions =
    code_header = textwrap.dedent("""
    from typing import List
    import torch
    """)
    cg_out(code_header)
    cg_right(code_header)

    cg_out.script_decorator()  # this is @torch.jit.script
    cg_out(textwrap.dedent(f"""
    def main(x1: torch.Tensor, x2: torch.Tensor, ws: torch.Tensor, w3j: torch.Tensor) -> torch.Tensor:
        {'x1, x2 = torch.broadcast_tensors(x1[..., :, None], x2[..., None, :])' if shared_weights else ''}
        {'x1, x2, ws = torch.broadcast_tensors(x1[..., :, None, None], x2[..., None, :, None], ws[..., None, None, :])' if not shared_weights else ''}
    """))
    cg_out.indent()

    cg_right.script_decorator()  # this is @torch.jit.script
    cg_right(textwrap.dedent(f"""
    def main(x2: torch.Tensor, ws: torch.Tensor, w3j: torch.Tensor) -> torch.Tensor:
        {'x2, ws = torch.broadcast_tensors(x2[..., :, None], ws[..., None, :])' if not shared_weights else ''}
    """))
    cg_right.indent()

    # = Short-circut for zero dimensional =
    if irreps_in1.dim == 0 or irreps_in2.dim == 0 or irreps_out.dim == 0:
        cg_out(f"return x1.new_zeros(x1.shape[{':-2' if shared_weights else ':-3'}] + ({irreps_out.dim},))")
        cg_right(f"return x2.new_zeros(x2.shape[{':-1' if shared_weights else ':-2'}] + ({irreps_in1.dim}, {irreps_out.dim},))")
        # Short circut
        # the empty list is wigners
        return cg_out, cg_right, []

    # = Broadcast inputs =
    # The if-else block is needed to avoid an internal TorchScript compiler bug related to the early return.
    cg_out(textwrap.dedent(f"""
        {'x1, x2 = x1[..., :, 0], x2[..., 0, :]' if shared_weights else ''}
        {'x1, x2, ws = x1[..., :, 0, 0], x2[..., 0, :, 0], ws[..., 0, 0, :]' if not shared_weights else ''}
        size = x1.shape[:-1]
        outsize = size + ({irreps_out.dim},)
        assert x1.shape[-1] == {irreps_in1.dim}, "Incorrect feature dimension for x1"
        assert x2.shape[-1] == {irreps_in2.dim}, "Incorrect feature dimension for x2"

        x1 = x1.reshape(-1, {irreps_in1.dim})
        x2 = x2.reshape(-1, {irreps_in2.dim})
        {f"ws = ws.reshape(-1, {weight_numel})" if weight_numel > 0 else ""}

        if x1.shape[0] == 0:
            return x1.new_zeros(outsize)
        else:
            batch = x1.shape[0]
    """))
    # ^ For forward(), we will accululate the various outputs independently
    # and then concatinate them: this improves performance by avoiding in-place operations,
    # which are bad for autograd.

    cg_right(textwrap.dedent(f"""
        {'x2, ws = x2[..., :, 0], ws[..., 0, :]' if not shared_weights else ''}
        size = x2.shape[:-1]
        outsize = size + ({irreps_in1.dim}, {irreps_out.dim},)
        assert x2.shape[-1] == {irreps_in2.dim}, "Incorrect feature dimension for x2"

        x2 = x2.reshape(-1, {irreps_in2.dim})
        {f"ws = ws.reshape(-1, {weight_numel})" if weight_numel > 0 else ""}

        if x2.shape[0] == 0:
            return x2.new_zeros(outsize)
        else:
            batch = x2.shape[0]
            out = x2.new_zeros((batch, {irreps_in1.dim}, {irreps_out.dim}))
    """))
    # ^ for right, it is not simple to collect different inputs independently,
    #   so we retain the in-place operations at a small performance cost.
    #   So, we allocate output space.

    # = Put everything in the else block =
    cg_out.indent()
    cg_right.indent()

    # = extract wigners =
    wigners = []

    # this function will only be called when we .generate the code generators, so it will have captured everything that gets added to `wigners`
    def code_wigners(profile=False):
        flat_wigner_index = 0
        code_wigners = []
        for i, (l_1, l_2, l_out) in enumerate(wigners):
            shape = (2 * l_1 + 1, 2 * l_2 + 1, 2 * l_out + 1)
            code_wigners.append(f"w3j_{i} = w3j[{flat_wigner_index}:{flat_wigner_index + prod(shape)}].reshape({tuple(shape)})")
            flat_wigner_index += prod(shape)
        return '\n'.join(code_wigners)

    cg_out(code_wigners)
    cg_right(code_wigners)

    # = extract individual input irreps =
    for i_1, (mul_1, (l_1, p_1)) in enumerate(irreps_in1):
        index_1 = irreps_in1[:i_1].dim
        dim_1 = mul_1 * (2 * l_1 + 1)
        cg_out(f"x1_{i_1} = x1[:, {index_1}:{index_1+dim_1}].reshape(batch, {mul_1}, {2 * l_1 + 1})")
    cg_out("\n")

    for i_2, (mul_2, (l_2, p_2)) in enumerate(irreps_in2):
        index_2 = irreps_in2[:i_2].dim
        dim_2 = mul_2 * (2 * l_2 + 1)
        line = f"x2_{i_2} = x2[:, {index_2}:{index_2+dim_2}].reshape(batch, {mul_2}, {2 * l_2 + 1})"
        cg_out(line)
        cg_right(line)
    cg_out("\n")
    cg_right("\n")

    z = '' if shared_weights else 'z'
    last_ss = None

    index_w = -1
    flat_weight_i = 0

    for ins_number, ins in enumerate(instructions):
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

        line = (
            f"with torch.autograd.profiler.record_function("
            f"'{irreps_in1[ins.i_in1:ins.i_in1+1]} x {irreps_in2[ins.i_in2:ins.i_in2+1]} "
            f"= {irreps_out[ins.i_out:ins.i_out+1]} {ins.connection_mode} {ins.has_weight}'):"
        )
        cg_out(line)
        cg_right(line)
        cg_out.indent()
        cg_right.indent()

        cg_out(f"s1 = x1_{ins.i_in1}")
        line = f"s2 = x2_{ins.i_in2}"
        cg_out(line)
        cg_right(line)

        cg_right(f"e1 = torch.eye({mul_1}, dtype=x2.dtype, device=x2.device)")

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
            line = f"ws_{index_w} = ws[:, {flat_weight_i}:{flat_weight_i + prod(ins.path_shape)}].reshape({(() if shared_weights else (-1,)) + tuple(ins.path_shape)})"
            cg_out(line)
            cg_right(line)
            flat_weight_i += prod(ins.path_shape)

        done: bool = True
        if specialized_code:
            # optimized code for special cases:
            # 0 x 0 = 0
            # 0 x L = L
            # L x 0 = L
            # L x L = 0
            # 1 x 1 = 1

            if (l_1, l_2, l_out) == (0, 0, 0) and ins.connection_mode in ['uvw', 'uvu'] and normalization in ['component', 'norm'] and ins.has_weight:
                cg_out(f"s1 = s1.reshape(batch, {mul_1})")
                line = f"s2 = s2.reshape(batch, {mul_2})"
                cg_out(line)
                cg_right(line)

                if ins.connection_mode == 'uvw':
                    cg_out.einsum(f"{z}uvw,zu,zv->zw", f"ws_{index_w}", "s1", "s2")
                    cg_right.einsum(f"{z}uvw,zv->zuw", f"ws_{index_w}", "s2")
                if ins.connection_mode == 'uvu':
                    cg_out.einsum(f"{z}uv,zu,zv->zu", f"ws_{index_w}", "s1", "s2")
                    cg_right.einsum(f"{z}uv,uw,zv->zuw", f"ws_{index_w}", "e1", "s2")

            elif l_1 == 0 and l_2 == l_out and ins.connection_mode in ['uvw', 'uvu'] and normalization == 'component' and ins.has_weight:
                cg_out(f"s1 = s1.reshape(batch, {mul_1})")

                if ins.connection_mode == 'uvw':
                    cg_out.einsum(f"{z}uvw,zu,zvi->zwi", f"ws_{index_w}", "s1", "s2")
                    cg_right.einsum(f"{z}uvw,zvi->zuwi", f"ws_{index_w}", "s2")
                if ins.connection_mode == 'uvu':
                    cg_out.einsum(f"{z}uv,zu,zvi->zui", f"ws_{index_w}", "s1", "s2")
                    cg_right.einsum(f"{z}uv,uw,zvi->zuwi", f"ws_{index_w}", "e1", "s2")

            elif l_1 == l_out and l_2 == 0 and ins.connection_mode in ['uvw', 'uvu'] and normalization == 'component' and ins.has_weight:
                cg_out(f"s2 = s2.reshape(batch, {mul_2})")
                cg_right(f"s2 = s2.reshape(batch, {mul_2})")
                cg_right(f"wig = torch.eye({2 * l_1 + 1}, dtype=x2.dtype, device=x2.device)")

                if ins.connection_mode == 'uvw':
                    cg_out.einsum(f"{z}uvw,zui,zv->zwi", f"ws_{index_w}", "s1", "s2")
                    cg_right.einsum(f"{z}uvw,ij,zv->zuiwj", f"ws_{index_w}", "wig", "s2")
                if ins.connection_mode == 'uvu':
                    cg_out.einsum(f"{z}uv,zui,zv->zui", f"ws_{index_w}", "s1", "s2")
                    cg_right.einsum(f"{z}uv,ij,uw,zv->zuiwj", f"ws_{index_w}", "wig", "e1", "s2")

            elif l_1 == l_2 and l_out == 0 and ins.connection_mode == 'uvw' and normalization == 'component' and ins.has_weight:
                # Cl_l_0 = eye / sqrt(2L+1)
                cg_out.einsum(f"{z}uvw,zui,zvi->zw", f"ws_{index_w}", "s1", "s2", div_consts=sqrt(2 * l_1 + 1))
                cg_right.einsum(f"{z}uvw,zvi->zuiw", f"ws_{index_w}", "s2", div_consts=sqrt(2 * l_1 + 1))

            elif l_1 == l_2 and l_out == 0 and ins.connection_mode == 'uvu' and normalization == 'component' and ins.has_weight:
                # Cl_l_0 = eye / sqrt(2L+1)
                cg_out.einsum(f"{z}uv,zui,zvi->zu", f"ws_{index_w}", "s1", "s2", div_consts=sqrt(2 * l_1 + 1))
                cg_right.einsum(f"{z}uv,uw,zvi->zuiw", f"ws_{index_w}", "e1", "s2", div_consts=sqrt(2 * l_1 + 1))

            elif l_1 == l_2 and l_out == 0 and ins.connection_mode == 'uuu' and normalization == 'component' and ins.has_weight:
                # Cl_l_0 = eye / sqrt(2L+1)
                cg_out.einsum(f"{z}u,zui,zui->zu", f"ws_{index_w}", "s1", "s2", div_consts=sqrt(2 * l_1 + 1))
                cg_right.einsum(f"{z}u,uw,zui->zuiw", f"ws_{index_w}", "e1", "s2", div_consts=sqrt(2 * l_1 + 1))

            elif l_1 == l_2 and l_out == 0 and ins.connection_mode == 'uuu' and normalization == 'component' and not ins.has_weight:
                # Cl_l_0 = eye / sqrt(2L+1)
                cg_out.einsum("zui,zui->zu", "s1", "s2", div_consts=sqrt(2 * l_1 + 1))
                cg_right.einsum("uw,zui->zuiw", "e1", "s2", div_consts=sqrt(2 * l_1 + 1))

            elif (l_1, l_2, l_out) == (1, 1, 1) and ins.connection_mode == 'uvw' and normalization == 'component' and ins.has_weight:
                # C1_1_1 = levi-civita / sqrt(2)
                cg_out(f"s1 = s1.reshape(batch, {mul_1}, 1, {2 * l_1 + 1})")
                cg_out(f"s2 = s2.reshape(batch, 1, {mul_2}, {2 * l_2 + 1})")
                cg_out("s1, s2 = torch.broadcast_tensors(s1, s2)")
                cg_out("s1xs2 = torch.cross(s1, s2, dim=3)")
                cg_out.einsum(f"{z}uvw,zuvi->zwi", f"ws_{index_w}", "s1xs2", div_consts=sqrt(2))

                if (l_1, l_2, l_out) in wigners:
                    index_w3j = wigners.index((l_1, l_2, l_out))
                else:
                    index_w3j = len(wigners)
                    wigners += [(l_1, l_2, l_out)]

                cg_right.einsum(f"{z}uvw,ijk,zvj->zuiwk", f"ws_{index_w}", f"w3j_{index_w3j}", "s2")

            elif (l_1, l_2, l_out) == (1, 1, 1) and ins.connection_mode == 'uvu' and normalization == 'component' and ins.has_weight:
                # C1_1_1 = levi-civita / sqrt(2)
                cg_out(f"s1 = s1.reshape(batch, {mul_1}, 1, {2 * l_1 + 1})")
                cg_out(f"s2 = s2.reshape(batch, 1, {mul_2}, {2 * l_2 + 1})")
                cg_out("s1, s2 = torch.broadcast_tensors(s1, s2)")
                cg_out("s1xs2 = torch.cross(s1, s2, dim=3)")
                cg_out.einsum(f"{z}uv,zuvi->zui", f"ws_{index_w}", "s1xs2", div_consts=sqrt(2))

                if (l_1, l_2, l_out) in wigners:
                    index_w3j = wigners.index((l_1, l_2, l_out))
                else:
                    index_w3j = len(wigners)
                    wigners += [(l_1, l_2, l_out)]

                cg_right.einsum(f"{z}uv,ijk,uw,zvj->zuiwk", f"ws_{index_w}", f"w3j_{index_w3j}", "e1", "s2")

            else:
                # We didn't make a specialized version
                done = False
        # == end specialized code ==
        if not done:
            # We didn't make this instruction specialized, so do the general case
            if last_ss != (ins.i_in1, ins.i_in2, ins.connection_mode[:2]):
                if ins.connection_mode[:2] == 'uv':
                    cg_out.einsum('zui,zvj->zuvij', 's1', 's2', out_var='ss')
                if ins.connection_mode[:2] == 'uu':
                    cg_out.einsum('zui,zuj->zuij', 's1', 's2', out_var='ss')
                last_ss = (ins.i_in1, ins.i_in2, ins.connection_mode[:2])

            if (l_1, l_2, l_out) in wigners:
                index_w3j = wigners.index((l_1, l_2, l_out))
            else:
                index_w3j = len(wigners)
                wigners += [(l_1, l_2, l_out)]

            if ins.connection_mode == 'uvw':
                assert ins.has_weight
                cg_out.einsum(f"{z}uvw,ijk,zuvij->zwk", f"ws_{index_w}", f"w3j_{index_w3j}", "ss")
                cg_right.einsum(f"{z}uvw,ijk,zvj->zuiwk", f"ws_{index_w}", f"w3j_{index_w3j}", "s2")
            if ins.connection_mode == 'uvu':
                assert mul_1 == mul_out
                if ins.has_weight:
                    cg_out.einsum(f"{z}uv,ijk,zuvij->zuk", f"ws_{index_w}", f"w3j_{index_w3j}", "ss")
                    cg_right.einsum(f"{z}uv,ijk,uw,zvj->zuiwk", f"ws_{index_w}", f"w3j_{index_w3j}", "e1", "s2")
                else:
                    cg_out.einsum("ijk,zuvij->zuk", f"w3j_{index_w3j}", "ss")
                    cg_right.einsum("ijk,uw,zvj->zuiwk", f"w3j_{index_w3j}", "e1", "s2")
            if ins.connection_mode == 'uvv':
                assert mul_2 == mul_out
                if ins.has_weight:
                    cg_out.einsum(f"{z}uv,ijk,zuvij->zvk", f"ws_{index_w}", f"w3j_{index_w3j}", "ss")
                    cg_right.einsum(f"{z}uv,ijk,zvj->zuivk", f"ws_{index_w}", f"w3j_{index_w3j}", "s2")
                else:
                    cg_out.einsum("ijk,zuvij->zvk", f"w3j_{index_w3j}", "ss")
                    cg_right(f"s2ones = torch.ones({mul_1}, device=s2.device, dtype=s2.dtype)")
                    cg_right.einsum("u,ijk,zvj->zuivk", "s2ones", f"w3j_{index_w3j}", "s2")
            if ins.connection_mode == 'uuw':
                assert mul_1 == mul_2
                if ins.has_weight:
                    cg_out.einsum(f"{z}uw,ijk,zuij->zwk", f"ws_{index_w}", f"w3j_{index_w3j}", "ss")
                    cg_right.einsum(f"{z}uw,ijk,zuj->zuiwk", f"ws_{index_w}", f"w3j_{index_w3j}", "s2")
                else:
                    assert mul_out == 1
                    cg_out.einsum("ijk,zuij->zk", f"w3j_{index_w3j}", "ss")
                    cg_right.einsum("ijk,zuj->zuik", f"w3j_{index_w3j}", "s2")
            if ins.connection_mode == 'uuu':
                assert mul_1 == mul_2 == mul_out
                if ins.has_weight:
                    cg_out.einsum(f"{z}u,ijk,zuij->zuk", f"ws_{index_w}", f"w3j_{index_w3j}", "ss")
                    cg_right.einsum(f"{z}u,ijk,uw,zuj->zuiwk", f"ws_{index_w}", f"w3j_{index_w3j}", "e1", "s2")
                else:
                    cg_out.einsum("ijk,zuij->zuk", f"w3j_{index_w3j}", "ss")
                    cg_right.einsum("ijk,uw,zuj->zuiwk", f"w3j_{index_w3j}", "e1", "s2")
            if ins.connection_mode == 'uvuv':
                assert mul_1 * mul_2 == mul_out
                if ins.has_weight:
                    cg_out.einsum(f"{z}uv,ijk,zuvij->zuvk", f"ws_{index_w}", f"w3j_{index_w3j}", "ss")
                    cg_right.einsum(f"{z}uv,ijk,uw,zvj->zuiwvk", f"ws_{index_w}", f"w3j_{index_w3j}", "e1", "s2")
                else:
                    cg_out.einsum("ijk,zuvij->zuvk", f"w3j_{index_w3j}", "ss")
                    cg_right.einsum("ijk,uw,zvj->zuiwvk", f"w3j_{index_w3j}", "e1", "s2")

        # multiply with alpha
        cg_out.scalar_multiply("_ein_out", alpha, out_var="_ein_out")
        cg_right.scalar_multiply("_ein_out", alpha, out_var="_ein_out")
        # sum this instruction into outputs
        cg_out(f"out_ins_{ins_number} = _ein_out.reshape(batch, {dim_out})")
        cg_right(f"out[:, {index_1}:{index_1+dim_1}, {index_out}:{index_out+dim_out}] += _ein_out.reshape(batch, {dim_1}, {dim_out})")
        # Dedent out of the profiler block
        cg_out.dedent()
        cg_right.dedent()
        cg_out("\n")
        cg_right("\n")

    # = Return the result =
    # for forward(), we now sum the individual paths at the end:
    out_ir_strs = [
        " + ".join(
            f"out_ins_{ins_number}"
            for ins_number in range(len(instructions))
            if instructions[ins_number].i_out == ir_out_num
        )
        for ir_out_num in range(len(irreps_out))
    ]
    cg_out(f"return torch.cat(({', '.join(out_ir_strs)},), dim=-1).reshape(outsize)")
    # for right(), we just reshape and return
    cg_right("return out.reshape(outsize)")

    return cg_out, cg_right, wigners
