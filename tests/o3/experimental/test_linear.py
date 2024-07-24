import e3nn_jax
import jax
import numpy as np
import torch
from e3nn import o3
import pytest


@pytest.mark.parametrize(
    "irreps_in,irreps_out",
    [
        ("2x0e + 2x1o", "3x0e + 3x1o"),
        ("1x0e + 2x1o", "3x0e + 3x1o"),
        ("1x0e + 2x1o", "3x0e + 3x1o + 1x2e"),
        ("1x0e + 2x1o + 1x2e", "3x0e + 3x1o"),
    ],
)
def test_linear(irreps_in, irreps_out):
    input = e3nn_jax.normal(irreps_in, jax.random.PRNGKey(0))
    linear = e3nn_jax.flax.Linear(
        irreps_in=irreps_in,
        irreps_out=irreps_out,
    )
    weight = linear.init(jax.random.PRNGKey(0), input)
    output = linear.apply(weight, input)

    weight_c = np.concatenate([w.ravel() for w in weight["params"].values()])
    linear = o3.experimental.Linearv2(irreps_in, irreps_out)
    output_torch = linear(torch.from_numpy(np.asarray(input.array)), torch.from_numpy(weight_c))
    assert np.allclose(output_torch.numpy(), output.array, rtol=1e-5, atol=1e-6)
