import argparse
import logging

import torch
from torch.utils.benchmark import Timer

from e3nn.o3 import Irreps, FullyConnectedTensorProduct, ElementwiseTensorProduct
from e3nn.util.jit import compile


logging.basicConfig(level=logging.DEBUG)


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
    parser.add_argument("--irreps", type=str, default="8x0e + 8x1e + 8x2e + 8x3o")
    parser.add_argument("--irreps-in1", type=str, default=None)
    parser.add_argument("--irreps-in2", type=str, default=None)
    parser.add_argument("--irreps-out", type=str, default=None)
    parser.add_argument("--cuda", type=t_or_f, default=True)
    parser.add_argument("--backward", type=int, default=1)
    parser.add_argument("--opt-ein", type=t_or_f, default=True)
    parser.add_argument("--specialized-code", type=t_or_f, default=True)
    parser.add_argument("--explicit-backward", type=t_or_f, default=False)
    parser.add_argument("--elementwise", action='store_true')
    parser.add_argument("-n", type=int, default=1000)
    parser.add_argument("--batch", type=int, default=10)

    args = parser.parse_args()

    device = 'cuda' if (torch.cuda.is_available() and args.cuda) else 'cpu'
    args.cuda = device == 'cuda'

    print("======= Benchmark with settings: ======")
    for key, val in vars(args).items():
        print(f"{key:>18} : {val}")
    print("="*40)

    irreps_in1 = Irreps(args.irreps_in1 if args.irreps_in1 else args.irreps)
    irreps_in2 = Irreps(args.irreps_in2 if args.irreps_in2 else args.irreps)
    irreps_out = Irreps(args.irreps_out if args.irreps_out else args.irreps)

    compile_opts = dict(
        specialized_code=args.specialized_code,
        optimize_einsums=args.opt_ein,
        explicit_backward=args.explicit_backward
    )

    if args.elementwise:
        tp = ElementwiseTensorProduct(
            irreps_in1,
            irreps_in2,
            compile_options=compile_opts
        )
        if args.backward:
            print("Elementwise TP has no weights, cannot backward. Setting --backward False.")
            args.backward = False
    else:
        tp = FullyConnectedTensorProduct(
            irreps_in1,
            irreps_in2,
            irreps_out,
            compile_options=compile_opts
        )
    tp = tp.to(device=device)
    assert len(tp.instructions) > 0, "Bad irreps, no instructions"
    print(f"Tensor product: {tp}")
    print("Instructions:")
    for ins in tp.instructions:
        print(f"  {ins}")

    # from https://pytorch.org/docs/master/_modules/torch/utils/benchmark/utils/timer.html#Timer.timeit
    warmup = max(int(args.n // 100), 1)

    inputs = [
        (
            irreps_in1.randn(args.batch, -1).to(device=device),
            irreps_in2.randn(args.batch, -1).to(device=device)
        )
        for _ in range(args.n + warmup)
    ]

    # compile
    if args.jit:
        tp = compile(tp)

    print("starting...")

    if args.backward == 0:
        extra = ""
    elif args.backward == 1:
        extra = "out.sum().backward()\n"
    elif args.backward == 2:
        for x1, x2 in inputs:
            x1.requires_grad_(True)
            x2.requires_grad_(True)
        extra = (
            "grad = torch.autograd.grad(out.sum(), x1, create_graph=True)[0]\n"
            "grad.sum().backward()\n"
        )
    else:
        raise ValueError(f"Invalid backward=`{args.backward}`>2")

    inputs = iter(inputs)

    t = Timer(
        stmt=(
            "tp.zero_grad()\n"
            "x1,x2 = next(inputs)\n"
            "out = tp(x1, x2)\n" + extra
        ),
        globals={'tp': tp, 'inputs': inputs}
    )

    perloop = t.timeit(args.n)

    print()
    print(perloop)


if __name__ == '__main__':
    main()
