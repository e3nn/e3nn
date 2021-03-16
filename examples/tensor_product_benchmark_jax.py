import argparse
import logging
logging.basicConfig(level=logging.DEBUG)

import torch
from torch.utils.benchmark import Timer
import numpy as np
from e3nn.util import prod

from e3nn.o3 import Irreps
from e3nn.o3.tensor_product.jax import tensor_product
from e3nn.o3.tensor_product import Instruction
import jax


# https://stackoverflow.com/a/15008806/1008938
def t_or_f(arg):
    ua = str(arg).upper()
    if 'TRUE'.startswith(ua):
       return True
    elif 'FALSE'.startswith(ua):
       return False
    else:
       raise ValueError(str(arg))


def main():
    parser = argparse.ArgumentParser(
        prog="tensor_product_benchmark"
    )
    parser.add_argument("--jit", type=t_or_f, default=True)
    parser.add_argument("--irreps-in1", type=str, default="8x1e + 8x2e + 8x3o")
    parser.add_argument("--irreps-in2", type=str, default="8x1e + 8x2e + 8x3o")
    parser.add_argument("--irreps-out", type=str, default="8x1e + 8x2e + 8x3o")
    parser.add_argument("--cuda", type=t_or_f, default=True)
    parser.add_argument("--backward", type=t_or_f, default=True)
    parser.add_argument("--opt-ein", type=t_or_f, default=True)
    parser.add_argument("--specialized-code", type=t_or_f, default=True)
    parser.add_argument("-n", type=int, default=1000)
    parser.add_argument("--batch", type=int, default=10)

    args = parser.parse_args()

    device = 'cuda' if (torch.cuda.is_available() and args.cuda) else 'cpu'
    args.cuda = device == 'cuda'

    print("======= Benchmark with settings: ======")
    for key, val in vars(args).items():
        print(f"{key:>18} : {val}")
    print("="*40)

    irreps_in1 = Irreps(args.irreps_in1)
    irreps_in2 = Irreps(args.irreps_in2)
    irreps_out = Irreps(args.irreps_out)

    instructions = [
        Instruction(i_1, i_2, i_out, 'uvw', True, 1.0, (mul_1, mul_2, mul_out))
        for i_1, (mul_1, ir_1) in enumerate(irreps_in1)
        for i_2, (mul_2, ir_2) in enumerate(irreps_in2)
        for i_out, (mul_out, ir_out) in enumerate(irreps_out)
        if ir_out in ir_1 * ir_2
    ]

    tp = tensor_product(
        irreps_in1,
        [1.0 for _ in irreps_in1],
        irreps_in2,
        [1.0 for _ in irreps_in2],
        irreps_out,
        [1.0 for _ in irreps_out],
        instructions=instructions,
        shared_weights=True,
        specialized_code=args.specialized_code,
        optimize_einsums=args.opt_ein
    )

    def f(x, y, w, w3j):
        return tp(x, y, w, w3j).sum()

    dtp = jax.grad(f, 2)

    # from https://pytorch.org/docs/master/_modules/torch/utils/benchmark/utils/timer.html#Timer.timeit
    warmup = max(int(args.n // 100), 1)

    inputs = iter([
        (
            irreps_in1.randn(args.batch, -1).numpy(),
            irreps_in2.randn(args.batch, -1).numpy(),
            np.random.randn(sum(prod(ins.path_shape) for ins in instructions if ins.has_weight)),
            np.random.randn(5000),
        )
        for _ in range(args.n + warmup)
    ])

    # compile
    if args.jit:
        tp = jax.jit(tp)
        dtp = jax.jit(dtp)

    print("starting...")

    t = Timer(
        stmt=(
            "dtp(*next(inputs))\n" if args.backward else "tp(*next(inputs))\n"
        ),
        globals={'tp': tp, 'dtp': dtp, 'inputs': inputs}
    )

    perloop = t.timeit(args.n)

    print()
    print(perloop)


if __name__ == '__main__':
    main()
