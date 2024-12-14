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

    LMAX = 3
    CHANNEL = 128
    BATCH = 100

    for lmax in range(LMAX, LMAX + 1):
        irreps = o3.Irreps.spherical_harmonics(lmax)
        irreps_x = (CHANNEL * irreps).regroup()
        x = irreps_x.randn(BATCH, -1).to(device=device)
        print(f"{irreps_x} \otimes {irreps_x}")

        tp = o3.TensorSquare(irreps_x)  # Doesnt work with fullgraph=True

        tp_compile = torch.compile(tp, mode=compile_mode).to(device=device)

        print(
            f"TP Torch 2.0 lmax {lmax} channel {CHANNEL} batch {BATCH}: {print_performance(lambda: tp_compile(x), times=100, repeat=10)*1000:.3f}ms"
        )

        tp_experimental = o3.experimental.TensorSquarev2(irreps_x)
        tp_experimental_compile = torch.compile(tp_experimental, mode=compile_mode, fullgraph=True).to(device=device)
        print(
            f"TP Experimental Torch 2.0 lmax {lmax} channel {CHANNEL} batch {BATCH}: {print_performance(lambda: tp_experimental_compile(x), times=100, repeat=10)*1000:.3f}ms"
        )


if __name__ == "__main__":
    main()
