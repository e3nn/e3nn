# flake8: noqa


def run_benchmark(irreps_type):
    import torch
    from torch._inductor.utils import print_performance

    # Borrowed from https://github.com/pytorch-labs/gpt-fast/blob/db7b273ab86b75358bd3b014f1f022a19aba4797/generate.py#L16-L18
    torch.set_float32_matmul_precision("high")
    import torch._dynamo.config
    import torch._inductor.config

    torch._inductor.config.coordinate_descent_tuning = True
    torch._inductor.config.triton.unique_kernel_names = True

    device = "cuda"
    compile_mode = "max-autotune-no-cudagraphs"  # Use reduce-overhead for quick compile times

    from e3nn import o3, experimental
    import numpy as np
    from torch import nn
    import time

    LMAX = 8
    CHANNEL = 128
    BATCH = 100

    module = o3

    for lmax in range(1, LMAX + 1):
        irreps = module.Irreps.spherical_harmonics(lmax)

        irreps_in = (CHANNEL * irreps).regroup()
        x = torch.randn(irreps_in.dim).to(device=device)
        irreps_out = (int(CHANNEL / 2) * irreps).regroup()
        print(f"{irreps_type.upper()}: {irreps_in} -> {irreps_out}")

        linear = module.Linear(irreps_in, irreps_out)
        linear_compile = torch.compile(linear, backend=compile_mode).to(device=device)
        print(
            f"{irreps_type.upper()} Linear lmax {lmax} channel {CHANNEL} batch {BATCH}: {print_performance(lambda: linear_compile(x), times=100, repeat=10)*1000:.3f}ms"
        )

        linear_experimental = experimental.Linearv2(irreps_in, irreps_out)

        linear_experimental_compile = torch.compile(linear_experimental, backend=compile_mode, fullgraph=True).to(
            device=device
        )
        print(
            f"{irreps_type.upper()} Linear Experimental lmax {lmax} channel {CHANNEL} batch {BATCH}: {print_performance(lambda: linear_experimental_compile(x, linear.weight), times=100, repeat=10)*1000:.3f}ms"
        )


def main():
    for irreps_type in ["o3"]:
        print(f"\nRunning benchmark for {irreps_type.upper()}")
        run_benchmark(irreps_type)


if __name__ == "__main__":
    main()
