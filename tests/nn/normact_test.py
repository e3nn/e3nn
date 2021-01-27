import torch

import e3nn
from e3nn.nn import NormActivation

def test_norm_activation():
    irreps_in = e3nn.o3.Irreps("4x0e + 5x1o")
    N_batch = 3
    in_features = torch.randn(3, irreps_in.dim)
    # Set some features to zero to test avoiding divide by zero
    in_features[0, 0] = 0 # batch 0, scalar 0
    in_features[1, 4:4+3] = 0 # batch 0, vector 1

    norm_act = NormActivation(
        irreps_in=irreps_in,
        scalar_nonlinearity=torch.tanh,
    )

    # Assert that there are no biases
    assert len(list(norm_act.parameters())) == 0

    out = norm_act(in_features)
    for batch in range(N_batch):
        # scalars should be the nonlin of their abs with the same sign.
        scalar_in = in_features[batch, :4]
        assert torch.allclose(
            torch.sign(scalar_in)*torch.tanh(scalar_in.abs()), # tanh of in norm
            out[batch, :4]
        )
        # vectors
        # first, check norms:
        vector_in = in_features[batch, 4:].reshape(5, 3)
        in_norms = vector_in.norm(dim=-1)
        vector_out = out[batch, 4:].reshape(5, 3)
        out_norms = vector_out.norm(dim=-1)
        # Can only check direction on vectors that have one:
        dir_mask = (in_norms > 0) & (out_norms > 0)
        assert torch.allclose(torch.tanh(in_norms), out_norms)
        # then that directions are unchanged:
        assert torch.allclose(
            vector_in[dir_mask] / in_norms[dir_mask, None],
            vector_out[dir_mask] / out_norms[dir_mask, None]
        )
