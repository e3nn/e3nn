import argparse
import logging

import torch

from e3nn.o3 import Irreps, FullyConnectedTensorProduct
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
    parser.add_argument("--irreps-in1", type=str, default="8x0e + 8x1e + 8x2e + 8x3e")
    parser.add_argument("--irreps-in2", type=str, default="8x0e + 8x1e + 8x2e + 8x3e")
    parser.add_argument("--irreps-out", type=str, default="8x0e + 8x1e + 8x2e + 8x3e")
    parser.add_argument("--cuda", type=t_or_f, default=True)
    parser.add_argument("--backward", type=t_or_f, default=True)
    parser.add_argument("--opt-ein", type=t_or_f, default=True)
    parser.add_argument("--specialized-code", type=t_or_f, default=True)
    parser.add_argument("-w", type=int, default=10)
    parser.add_argument("-n", type=int, default=3)
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
    compile_opts = dict(
        specialized_code=args.specialized_code,
        optimize_einsums=args.opt_ein
    )
    tp = FullyConnectedTensorProduct(
        irreps_in1,
        irreps_in2,
        irreps_out,
        compile_options=compile_opts
    )
    tp = tp.to(device=device)

    inputs = iter([
        (
            irreps_in1.randn(args.batch, -1).to(device=device),
            irreps_in2.randn(args.batch, -1).to(device=device)
        )
        for _ in range(1 + args.w + args.n)
    ])

    # compile
    if args.jit:
        tp = compile(tp)

    print("starting...")

    called_num = [0]

    def trace_handler(p):
        print(p.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))
        p.export_chrome_trace("test_trace_" + str(called_num[0]) + ".json")
        called_num[0] += 1

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(
            wait=1,
            warmup=args.w,
            active=args.n),
        on_trace_ready=trace_handler
    ) as p:
        for _ in range(1 + args.w + args.n):
            out = tp(*next(inputs))
            if args.backward:
                out.sum().backward()
            p.step()


if __name__ == '__main__':
    main()
