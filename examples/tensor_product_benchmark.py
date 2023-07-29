import argparse
import logging

import torch
from torch.utils.benchmark import Timer

from e3nn.o3 import Irreps, FullyConnectedTensorProduct, ElementwiseTensorProduct
from e3nn.util.jit import compile


logging.basicConfig(level=logging.DEBUG)


# https://stackoverflow.com/a/15008806/1008938
def t_or_f(arg) -> bool:
    ua = str(arg).upper()
    if "TRUE".startswith(ua):
        return True
    elif "FALSE".startswith(ua):
        return False
    else:
        raise ValueError(str(arg))


def main() -> None:
    parser = argparse.ArgumentParser(prog="tensor_product_benchmark")
    parser.add_argument("--jit", type=t_or_f, default=True)
    parser.add_argument("--irreps", type=str, default="8x0e + 8x1e + 8x2e + 8x3o")
    parser.add_argument("--irreps-in1", type=str, default=None)
    parser.add_argument("--irreps-in2", type=str, default=None)
    parser.add_argument("--irreps-out", type=str, default=None)
    parser.add_argument("--cuda", type=t_or_f, default=True)
    parser.add_argument("--backward", type=t_or_f, default=True)
    parser.add_argument("--opt-ein", type=t_or_f, default=True)
    parser.add_argument("--specialized-code", type=t_or_f, default=True)
    parser.add_argument("--elementwise", action="store_true")
    parser.add_argument("-n", type=int, default=1000)
    parser.add_argument("--batch", type=int, default=10)

    args = parser.parse_args()

    device = "cuda" if (torch.cuda.is_available() and args.cuda) else "cpu"
    args.cuda = device == "cuda"

    print("======= Benchmark with settings: ======")
    for key, val in vars(args).items():
        print(f"{key:>18} : {val}")
    print("=" * 40)

    irreps_in1 = Irreps(args.irreps_in1 if args.irreps_in1 else args.irreps)
    irreps_in2 = Irreps(args.irreps_in2 if args.irreps_in2 else args.irreps)
    irreps_out = Irreps(args.irreps_out if args.irreps_out else args.irreps)

    if args.elementwise:
        tp = ElementwiseTensorProduct(
            irreps_in1, irreps_in2, _specialized_code=args.specialized_code, _optimize_einsums=args.opt_ein
        )
        if args.backward:
            print("Elementwise TP has no weights, cannot backward. Setting --backward False.")
            args.backward = False
    else:
        tp = FullyConnectedTensorProduct(
            irreps_in1, irreps_in2, irreps_out, _specialized_code=args.specialized_code, _optimize_einsums=args.opt_ein
        )
    tp = tp.to(device=device)
    assert len(tp.instructions) > 0, "Bad irreps, no instructions"
    print(f"Tensor product: {tp}")
    print("Instructions:")
    for ins in tp.instructions:
        print(f"  {ins}")

    # from https://pytorch.org/docs/master/_modules/torch/utils/benchmark/utils/timer.html#Timer.timeit
    warmup = max(int(args.n // 100), 1)

    inputs = iter(
        [
            (irreps_in1.randn(args.batch, -1).to(device=device), irreps_in2.randn(args.batch, -1).to(device=device))
            for _ in range(args.n + warmup)
        ]
    )

    # compile
    if args.jit:
        tp = compile(tp)

    print("starting...")

    # tanh() forces it to realize the grad as a full size matrix rather than expanded (stride 0) ones
    t = Timer(
        stmt=("tp.zero_grad()\n" "out = tp(*next(inputs))\n" + ("out.tanh().sum().backward()\n" if args.backward else "")),
        globals={"tp": tp, "inputs": inputs},
    )

    perloop = t.timeit(args.n)

    print()
    print(perloop)


if __name__ == "__main__":
    main()
