# flake8: noqa


def main():
    import torch
    from torch._inductor.utils import print_performance

    # Borrowed from https://github.com/pytorch-labs/gpt-fast/blob/db7b273ab86b75358bd3b014f1f022a19aba4797/generate.py#L16-L18
    torch.set_float32_matmul_precision("high")
    import torch._dynamo.config
    import torch._inductor.config

    torch._inductor.config.coordinate_descent_tuning = True
    torch._inductor.config.triton.unique_kernel_names = True

    device = "cuda"
    compile_mode = (
        "max-autotune"  # Bringing out all of the tricks that Torch 2.0 has but "reduce-overhead" should work as well
    )

    from e3nn import o3, util
    import numpy as np
    from torch import nn
    import time

    LMAX = 8
    CHANNEL = 128
    BATCH = 100

    for lmax in range(1, LMAX + 1):
        irreps = o3.Irreps.spherical_harmonics(lmax)
        irreps_x = (CHANNEL * irreps).regroup()
        x = irreps_x.randn(BATCH, -1).to(device=device)
        irreps_y = irreps
        y = irreps_y.randn(BATCH, -1).to(device=device)
        print(f"{irreps_x} \otimes {irreps_y}")

        tp = o3.FullTensorProduct(irreps_x, irreps_y)  # Doesnt work with fullgraph=True

        tp_jit_compile = util.jit.compile(tp).to(device=device)

        tp_compile = torch.compile(tp, mode=compile_mode).to(device=device)
        print(
            f"TP JIT lmax {lmax} channel {CHANNEL} batch {BATCH}: {print_performance(lambda: tp_jit_compile(x, y), times=100, repeat=10)*1000:.3f}ms"
        )

        print(
            f"TP Torch 2.0 lmax {lmax} channel {CHANNEL} batch {BATCH}: {print_performance(lambda: tp_compile(x, y), times=100, repeat=10)*1000:.3f}ms"
        )

        tp_experimental = o3.experimental.FullTensorProductv2(irreps_x, irreps_y)
        tp_experimental_compile = torch.compile(tp_experimental, mode=compile_mode, fullgraph=True).to(device=device)
        print(
            f"TP Experimental Torch 2.0 lmax {lmax} channel {CHANNEL} batch {BATCH}: {print_performance(lambda: tp_experimental_compile(x, y), times=100, repeat=10)*1000:.3f}ms"
        )


if __name__ == "__main__":
    main()
