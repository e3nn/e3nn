import torch

from e3nn.io import CartesianTensor


def test_cartesian_tensor_cache():
    ct1 = CartesianTensor("ij=-ji")
    ct1_2 = CartesianTensor("ij=-ji")
    x = torch.randn(3, 3, 3)
    out1 = ct1.from_cartesian(x)
    assert torch.allclose(out1, ct1_2.from_cartesian(x))
    CartesianTensor.reset_rtp_cache()
    ct1_3 = CartesianTensor("ij=-ji")
    assert torch.allclose(out1, ct1_3.from_cartesian(x))

    ct2 = CartesianTensor("ijk=-jik=-ikj")
    ct2_2 = CartesianTensor("ijk=-jik=-ikj")
    # this actually calls rtp
    vectors = [torch.randn(3), torch.randn(3), torch.randn(3)]
    assert torch.allclose(ct2.from_vectors(*vectors), ct2_2.from_vectors(*vectors))


def test_cartesian_tensor():
    x = CartesianTensor("ij=ji")
    t = torch.arange(9).to(torch.float).view(3, 3)
    y = x.from_cartesian(t)
    z = x.to_cartesian(y)
    assert torch.allclose(z, (t + t.T)/2, atol=1e-5)
