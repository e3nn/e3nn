import torch

from e3nn import rs
from e3nn import tensor_block_cuda

from itertools import accumulate


def repeat_m(input, Rs=None, L_list=None, mul_sizes=None, input_offsets=None, output_offsets=None, batch_dim_first=True):
    assert Rs is not None or (L_list is not None and mul_sizes is not None), "Supply either representations (Rs) or lists with rotation orders (L_list) and multiplicities (mul_sizes)"

    L_list = L_list if L_list is not None else input.new_tensor(rs.extract_l(Rs), dtype=torch.int32)
    mul_sizes = mul_sizes if mul_sizes is not None else input.new_tensor(rs.extract_mul(Rs), dtype=torch.int32)

    input_offsets = input_offsets if input_offsets is not None else input.new_tensor([0] + list(accumulate(mul_sizes)), dtype=torch.int32)
    output_offsets = output_offsets if output_offsets is not None else input.new_tensor([0] + list(accumulate([mul * (2 * L + 1) for (mul, L) in zip(mul_sizes, L_list)])), dtype=torch.int32)

    if batch_dim_first:
        input = input.t().contiguous()
        output = tensor_block_cuda.repeat_m(input, L_list, mul_sizes, output_offsets, input_offsets)
        output = output.t().contiguous()
    else:
        output = tensor_block_cuda.repeat_m(input, L_list, mul_sizes, output_offsets, input_offsets)
    return output


def sum_m(input, Rs=None, L_list=None, mul_sizes=None, input_offsets=None, output_offsets=None, batch_dim_first=True):
    assert Rs is not None or (L_list is not None and mul_sizes is not None), "Supply either representations (Rs) or lists with rotation orders (L_list) and multiplicities (mul_sizes)"

    L_list = L_list if L_list is not None else input.new_tensor(rs.extract_l(Rs), dtype=torch.int32)
    mul_sizes = mul_sizes if mul_sizes is not None else input.new_tensor(rs.extract_mul(Rs), dtype=torch.int32)

    input_offsets = input_offsets if input_offsets is not None else input.new_tensor([0] + list(accumulate([mul * (2*L + 1) for (mul, L) in zip(mul_sizes, L_list)])), dtype=torch.int32)
    output_offsets = output_offsets if output_offsets is not None else input.new_tensor([0] + list(accumulate(mul_sizes)), dtype=torch.int32)

    if batch_dim_first:
        input = input.t().contiguous()
        output = tensor_block_cuda.sum_m(input, L_list, mul_sizes, output_offsets, input_offsets)
        output = output.t().contiguous()
    else:
        output = tensor_block_cuda.sum_m(input, L_list, mul_sizes, output_offsets, input_offsets)
    return output
