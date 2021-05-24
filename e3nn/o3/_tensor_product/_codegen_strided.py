from math import sqrt
from typing import List, Tuple

import torch
from torch import fx

from e3nn import o3
from e3nn.util import prod

from ._instruction import Instruction


def codegen_tensor_product_strided(
    irreps_in1: o3.StridedIrreps,
    in1_var: List[float],
    irreps_in2: o3.StridedIrreps,
    in2_var: List[float],
    irreps_out: o3.StridedIrreps,
    out_var: List[float],
    instructions: List[Instruction],
    normalization: str = 'component',
    shared_weights: bool = False,
    specialized_code: bool = True,
    optimize_einsums: bool = True,
) -> Tuple[fx.GraphModule, fx.GraphModule]:
    graph_out = fx.Graph()

    # Make a placeholder graph for right()
    graph_right = fx.Graph()
    graph_right.placeholder('x2', torch.Tensor)
    graph_right.placeholder('w', torch.Tensor)
    graph_right.call_function(
        torch._assert,
        args=(False, "Strided does not support right()")
    )
    graphmod_right = fx.GraphModule({}, graph_right, "tp_right")
    del graph_right

    # = Function definitions =
    # z u i
    x1s_out = fx.Proxy(graph_out.placeholder('x1', torch.Tensor))
    # z v j
    x2s_out = fx.Proxy(graph_out.placeholder('x2', torch.Tensor))
    # [z] p [uvw|uv|u]
    ws_out = fx.Proxy(graph_out.placeholder('w', torch.Tensor))

    batch_out = x1s_out.shape[0]

    # = Short-circut for zero dimensional =
    # We produce no code for empty instructions
    instructions = [ins for ins in instructions if 0 not in ins.path_shape]

    if len(instructions) == 0:
        out_out = x1s_out.new_zeros((batch_out,) + irreps_out.dim)

        graph_out.output(out_out.node, torch.Tensor)
        # Short circut
        return (
            fx.GraphModule({}, graph_out, "tp_forward"),
            graphmod_right
        )

    # = Determine number of weights and reshape weights ==
    path_shape = instructions[0].path_shape
    connection_mode = instructions[0].connection_mode
    has_weight = instructions[0].has_weight
    assert all(ins.path_shape == path_shape for ins in instructions)
    assert all(ins.connection_mode == connection_mode for ins in instructions)
    assert all(ins.has_weight == has_weight for ins in instructions)

    # all paths have the same shape:
    if has_weight:
        ws_out = ws_out.reshape(
            (tuple() if shared_weights else (-1,)) + (len(instructions),) + path_shape
        )  # [z] p uvw

    # [k] [(p)ij]
    big_w3j_shape = (
        irreps_out.base_irreps.dim,
        (
            irreps_in1.base_irreps.dim * irreps_in2.base_irreps.dim * (len(instructions) if has_weight else 1)
        )
    )
    big_w3j_indexes = []
    big_w3j_values = []

    for ins_i, ins in enumerate(instructions):
        mul_ir_in1 = irreps_in1[ins.i_in1]
        mul_ir_in2 = irreps_in2[ins.i_in2]
        mul_ir_out = irreps_out[ins.i_out]

        assert mul_ir_in1.ir.p * mul_ir_in2.ir.p == mul_ir_out.ir.p
        assert abs(mul_ir_in1.ir.l - mul_ir_in2.ir.l) <= mul_ir_out.ir.l <= mul_ir_in1.ir.l + mul_ir_in2.ir.l

        if mul_ir_in1.dim == 0 or mul_ir_in2.dim == 0 or mul_ir_out.dim == 0:
            continue

        alpha = ins.path_weight * out_var[ins.i_out] / sum(in1_var[i.i_in1] * in2_var[i.i_in2] for i in instructions if i.i_out == ins.i_out)

        alpha = sqrt(alpha / {
            'uvw': (mul_ir_in1.mul * mul_ir_in2.mul),
            'uvu': mul_ir_in2.mul,
            'uvv': mul_ir_in1.mul,
            'uuw': mul_ir_in1.mul,
            'uuu': 1,
        }[connection_mode])

        # Fill big w3j
        this_w3j = o3.wigner_3j(mul_ir_in1.ir.l, mul_ir_in2.ir.l, mul_ir_out.ir.l)
        if normalization == 'component':
            alpha *= (2 * mul_ir_out.ir.l + 1) ** 0.5
        elif normalization == 'norm':
            alpha *= (2 * mul_ir_in1.ir.l + 1) ** 0.5 * (2 * mul_ir_in2.ir.l + 1) ** 0.5
        this_w3j *= alpha
        # ^ ijk
        this_index = this_w3j.nonzero()
        this_value = this_w3j[this_index[:, 0], this_index[:, 1], this_index[:,2]]
        # now, we need to make the indexes flat
        this_k_index = this_index[:, 2] + irreps_out.base_irreps[:ins.i_out].dim
        # Flattened ij:
        this_pij_index = (this_index[:, 0] + irreps_in1.base_irreps[:ins.i_in1].dim) * irreps_in2.base_dim + (this_index[:, 1] + irreps_in2.base_irreps[:ins.i_in2].dim)
        # do the path offset
        if has_weight:
            # stride ij
            this_pij_index += ins_i * irreps_in1.base_dim * irreps_in2.base_dim
        big_w3j_indexes.append(torch.vstack((this_k_index, this_pij_index)))
        big_w3j_values.append(this_value)

    # Build the sparse matrix
    big_w3j = torch.sparse_coo_tensor(
        indices=torch.cat(big_w3j_indexes, dim=1),
        values=torch.cat(big_w3j_values, dim=0),
        size=big_w3j_shape
    )
    del big_w3j_indexes
    del big_w3j_values

    # - Run actual einsum -
    big_w3j_proxy = fx.Proxy(graph_out.get_attr("big_w3j"))
    # The einsum string index to prepend to the weights if the weights are not shared and have a batch dimension
    z = '' if shared_weights else 'z'
    u = connection_mode[0]
    v = connection_mode[1]
    w = connection_mode[2]
    weight_label = {
        "uvw": "uvw",
        "uuu": "u",
        "uuv": "uv",
        "uvu": "uv",
    }[connection_mode]
    if has_weight:
        weighted_outer_product = torch.einsum(
            f"z{u}i,{z}p{weight_label},z{v}j->pij{w}z",
            x1s_out.reshape(-1, irreps_in1.mul, irreps_in1.base_dim),
            ws_out,
            x2s_out.reshape(-1, irreps_in2.mul, irreps_in2.base_dim)
        )
    else:
        weighted_outer_product = torch.einsum(
            f"z{u}i,z{v}j->ij{w}z",
            x1s_out.reshape(-1, irreps_in1.mul, irreps_in1.base_dim),
            x2s_out.reshape(-1, irreps_in2.mul, irreps_in2.base_dim)
        )
    # Contract with the sparse w3j:
    # (p)ijwz -> [(p)ij][wz]
    weighted_outer_product = weighted_outer_product.reshape(
        big_w3j_shape[-1], -1
    )
    # [k][wz]:
    result = torch.mm(big_w3j_proxy, weighted_outer_product)
    # z[wk]
    result = result.reshape(irreps_out.base_dim, irreps_out.mul, -1).transpose(0, 2).reshape(-1, irreps_out.dim)

    graph_out.output(result.node, torch.Tensor)

    # check graphs
    graph_out.lint()

    graphmod_out = fx.GraphModule(
        {"big_w3j": big_w3j},
        graph_out,
        class_name="tp_forward"
    )

    # == Optimize ==
    # TODO: when eliminate_dead_code() is in PyTorch stable, use that
    if optimize_einsums:
        try:
            from opt_einsum_fx import optimize_einsums_full, jitable
        except ImportError:
            # opt_einsum_fx is not installed
            pass
        else:
            # TODO: true?
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
                irreps_in1.randn(batchdim, -1),
                irreps_in2.randn(batchdim, -1),
                torch.zeros(
                    1 if shared_weights else batchdim,
                    len(instructions) * prod(path_shape) * has_weight
                ),
            )

            graphmod_out = jitable(optimize_einsums_full(graphmod_out, example_inputs))

    return graphmod_out, graphmod_right
