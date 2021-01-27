import pytest

import torch

import e3nn
from e3nn.nn import NormActivation

@pytest.mark.parametrize('do_bias', [True, False])
def test_norm_activation(do_bias):
    nonlin = torch.sigmoid
    irreps_in = e3nn.o3.Irreps("4x0e + 5x1o")
    N_batch = 3
    in_features = torch.randn(N_batch, irreps_in.dim)
    # Set some features to zero to test avoiding divide by zero
    in_features[0, 0] = 0  # batch 0, scalar 0
    in_features[1, 4:4+3] = 0  # batch 0, vector 1

    norm_act = NormActivation(
        irreps_in=irreps_in,
        scalar_nonlinearity=nonlin,
        bias=do_bias
    )

    if do_bias:
        assert len(list(norm_act.parameters())) == 1
        norm_act.biases[:] = torch.randn(norm_act.biases.shape)
    else:
        # Assert that there are no biases
        assert len(list(norm_act.parameters())) == 0

    out = norm_act(in_features)

    if do_bias:
        assert out.requires_grad

    for batch in range(N_batch):
        # scalars should be the nonlin of their abs with the same sign.
        scalar_in = in_features[batch, :4]
        if do_bias:
            true_nonlin_arg = scalar_in.abs() + norm_act.biases[:4]
        else:
            true_nonlin_arg = scalar_in.abs()
        assert torch.allclose(
            torch.sign(scalar_in)*nonlin(true_nonlin_arg),
            out[batch, :4]
        )
        # vectors
        # first, check norms:
        vector_in = in_features[batch, 4:].reshape(5, 3)
        in_norms = vector_in.norm(dim=-1)
        vector_out = out[batch, 4:].reshape(5, 3)
        out_norms = vector_out.norm(dim=-1)
        # Can only check direction on vectors that have one:
        mask = (in_norms > 0) & (out_norms > 0)
        if do_bias:
            true_nonlin_arg = in_norms + norm_act.biases[4:]
        else:
            true_nonlin_arg = in_norms
        # Check norms for nonzero vectors
        assert torch.allclose(nonlin(true_nonlin_arg).abs()[mask], out_norms[mask])
        # Check that zeros maintained for zero inputs
        assert torch.allclose(in_norms[~mask], out_norms[~mask])
        # then that directions are unchanged:
        assert torch.allclose(
            vector_in[mask] / in_norms[mask, None],
            vector_out[mask] / out_norms[mask, None]
        )
