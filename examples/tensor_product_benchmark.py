import argparse
import logging

import torch
from torch.utils.benchmark import Timer

from e3nn.o3 import Irreps, StridedIrreps, TensorProduct
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
    parser.add_argument("--weighted", type=t_or_f, default=True)
    parser.add_argument("--strided", type=t_or_f, default=False)
    parser.add_argument("--cuda", type=t_or_f, default=True)
    parser.add_argument("--backward", type=int, default=1)
    parser.add_argument("--opt-ein", type=t_or_f, default=True)
    parser.add_argument("--specialized-code", type=t_or_f, default=True)
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

    if args.strided:
        irreps_in1, irreps_in2, irreps_out = (StridedIrreps(e) for e in (irreps_in1, irreps_in2, irreps_out))

    instr = [
        (i_1, i_2, i_out, 'uuu' if args.elementwise else 'uvw', args.weighted, 1.0)
        for i_1, (_, ir_1) in enumerate(irreps_in1)
        for i_2, (_, ir_2) in enumerate(irreps_in2)
        for i_out, (_, ir_out) in enumerate(irreps_out)
        if ir_out in ir_1 * ir_2
    ]
    tp = TensorProduct(
        irreps_in1,
        irreps_in2,
        irreps_out,
        instructions=instr,
        _specialized_code=args.specialized_code,
        _optimize_einsums=args.opt_ein
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

    if args.backward == 0:
        extra = ""
    elif args.backward == 1:
        extra = "out.sum().backward()\n"
        if not args.weighted:
            # need something to grad wrt
            for x1, x2 in inputs:
                x1.requires_grad_(True)
                x2.requires_grad_(True)
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

    print("starting...")

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
