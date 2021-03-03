"""Optimize einsums based on shapes for code generation."""

from typing import List, Optional

from opt_einsum.contract import contract_path

from e3nn.util import prod


# based on opt_einsum/backends/torch.py
class _CodeGenBackend:
    def __init__(self):
        self._tmp_idx = 0
        self.code = []

    def get_tmp(self):
        out = f"_ein_tmp{self._tmp_idx}"
        self._tmp_idx += 1
        return out

    def transpose(self, v, axes, out=None):
        axes = list(axes)
        if axes == list(range(len(axes))):
            # In this case, we're asking for the identity permutation, so just no-op
            if out is None:
                return v
            else:
                self.code.append(f"{out} = {v}  # identity permutation no-op")
        out = out if out is not None else self.get_tmp()
        self.code.append(f"{out} = {v}.permute({', '.join(str(a) for a in axes)})")
        return out

    def einsum(self, einstr, *args, out=None):
        out = out if out is not None else self.get_tmp()
        self.code.append(f"{out} = torch.einsum('{einstr}', {', '.join(args)})")
        return out

    def tensordot(self, x, y, axes=2, out=None):
        out = out if out is not None else self.get_tmp()
        self.code.append(f"{out} = torch.tensordot({x}, {y}, dims_self=[{', '.join(str(a) for a in axes[0])}], dims_other=[{', '.join(str(a) for a in axes[1])},])")
        return out

    def scalar_mul(self, x, mul: float):
        out = self.get_tmp()
        self.code.append(f"{out} = {x}.mul({mul})")
        return out

    def make_code(self):
        return "\n".join(self.code)


def opt_einsum_code(
    einstr: str,
    operands: List[str],
    arg_shapes: list,
    out_var: str,
    mul_const: Optional[float] = None,
    **opt_kwargs
):
    operands = list(operands)

    _, opt_path = contract_path(
        einstr,
        *arg_shapes,
        shapes=True,
        **opt_kwargs
    )

    cgb = _CodeGenBackend()

    # == deal with constants ==
    # TODO: check this
    # TODO: output too!
    # einsum is a linear operation in all operands, and as a result we
    # can multiply a scalar constant for the ouptu onto any of the inputs.
    # In terms of FLOPs, it's fastest then to do this multiplication with
    # the smallest of the operands
    if mul_const is not None:
        # which is smallest
        smallest = min((i for i in range(len(operands))), key=lambda i: prod(arg_shapes[i]))
        with_mul = cgb.scalar_mul(operands[smallest], mul_const)
        operands[smallest] = with_mul

    # setup for following
    contraction_list = opt_path.contraction_list
    # ===== Code based on opt_einsum's _core_contract ========
    for num, contraction in enumerate(contraction_list):
        inds, idx_rm, einsum_str, _, blas_flag = contraction

        tmp_operands = [operands.pop(x) for x in inds]

        # Do we need to deal with the output?
        handle_out = ((num + 1) == len(contraction_list))

        # Call tensordot (check if should prefer einsum)
        if blas_flag and ('EINSUM' not in blas_flag):
            # Checks have already been handled
            input_str, results_index = einsum_str.split('->')
            input_left, input_right = input_str.split(',')

            tensor_result = "".join(s for s in input_left + input_right if s not in idx_rm)

            if idx_rm:
                # Find indices to contract over
                left_pos, right_pos = [], []
                for s in idx_rm:
                    left_pos.append(input_left.find(s))
                    right_pos.append(input_right.find(s))

                # Construct the axes tuples in a canonical order
                axes = tuple(zip(*sorted(zip(left_pos, right_pos))))
            else:
                # Ensure axes is always pair of tuples
                axes = ((), ())

            # Contract!
            new_view = cgb.tensordot(*tmp_operands, axes=axes)

            # Build a new view if needed
            if (tensor_result != results_index) or handle_out:

                transpose = tuple(map(tensor_result.index, results_index))
                new_view = cgb.transpose(
                    new_view,
                    axes=transpose,
                    out=(out_var if handle_out else None)
                )

        # Call einsum
        else:
            # Do the contraction
            new_view = cgb.einsum(
                einsum_str,
                *tmp_operands,
                out=(out_var if handle_out else None)
            )

        # Append new items and dereference what we can
        operands.append(new_view)
        del tmp_operands, new_view
    # ==== end opt_einsum code =====

    outcode = cgb.make_code()
    return outcode, opt_path
