import argparse
import logging
logging.basicConfig(level=logging.DEBUG)

import torch
from torch.utils.benchmark import Timer

from e3nn.o3 import Irreps, FullyConnectedTensorProduct
from e3nn.util.jit import compile


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
    parser.add_argument("-n", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--batch", type=int, default=10)

    args = parser.parse_args()

    device = 'cuda' if (torch.cuda.is_available() and args.cuda) else 'cpu'
    args.cuda = device == 'cuda'

    print(" ===== Benchmark with settings: ===== ")
    for key, val in vars(args).items():
        print(f"{key:>15} : {val}")
    print("="*37)

    irreps_in1 = Irreps(args.irreps_in1)
    irreps_in2 = Irreps(args.irreps_in2)
    irreps_out = Irreps(args.irreps_out)
    tp = FullyConnectedTensorProduct(
        irreps_in1,
        irreps_in2,
        irreps_out,
    )
    tp = tp.to(device=device)

    inputs = iter([
        (
            irreps_in1.randn(args.batch, -1).to(device=device),
            irreps_in2.randn(args.batch, -1).to(device=device)
        )
        for _ in range(args.warmup + args.n + 1)]
    )

    # compile
    if args.jit:
        tp = compile(tp)

    # warmup
    for i in range(args.warmup):
        _ = tp(*next(inputs))

    print("starting...")

    t = Timer(
        stmt=(
            "tp.zero_grad()\n"
            "out = tp(*next(inputs))\n" + ("out.sum().backward()\n" if args.backward else '')
        ),
        globals={'tp': tp, 'inputs': inputs}
    )

    perloop = t.timeit(args.n)

    print()
    print(perloop)


if __name__ == '__main__':
    main()
