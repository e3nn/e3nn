import argparse
import time
import logging
logging.basicConfig(level=logging.DEBUG)

import torch
from e3nn.o3 import Irreps, FullyConnectedTensorProduct
from e3nn.util.jit import compile

parser = argparse.ArgumentParser(
    prog="tensor_product_benchmark"
)
parser.add_argument("--jit", type=bool, default=True)
parser.add_argument("--irreps-in1", type=str, default="8x1e + 8x2e + 8x3o")
parser.add_argument("--irreps-in2", type=str, default="8x1e + 8x2e + 8x3o")
parser.add_argument("--irreps-out", type=str, default="8x1e + 8x2e + 8x3o")
parser.add_argument("--cuda", type=bool, default=True)
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

inputs = iter([(irreps_in1.randn(args.batch, -1), irreps_in2.randn(args.batch, -1)) for _ in range(1 + args.warmup + args.n)])

# optimize (if that happens)
_ = tp(*next(inputs))

# compile
if args.jit:
    tp = compile(tp)

# warmup
for i in range(args.warmup):
    _ = tp(*next(inputs))

print("starting...")

start = time.time()

accumulate = torch.as_tensor(0.)

for input in inputs:
    tp.zero_grad()
    out = tp(*input)
    out.sum().backward()
    accumulate = accumulate + tp.weight[0]

end = time.time()

print(f"per-loop {(end - start) / args.n:0.5f} sec")