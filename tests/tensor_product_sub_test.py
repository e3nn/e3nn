import torch
from e3nn import o3
from e3nn.nn import FullyConnectedTensorProduct, Identity, Linear


def test_fully_connected():
    irreps_in1 = o3.Irreps("1e + 2e + 3x3o")
    irreps_in2 = o3.Irreps("1e + 2e + 3x3o")
    irreps_out = o3.Irreps("1e + 2e + 3x3o")
    m = FullyConnectedTensorProduct(irreps_in1, irreps_in2, irreps_out)
    print(m)
    m(torch.randn(irreps_in1.dim), torch.randn(irreps_in2.dim))


def test_id():
    irreps_in = o3.Irreps("1e + 2e + 3x3o")
    irreps_out = o3.Irreps("1e + 2e + 3x3o")
    m = Identity(irreps_in, irreps_out)
    print(m)
    m(torch.randn(irreps_in.dim))


def test_linear():
    irreps_in = o3.Irreps("1e + 2e + 3x3o")
    irreps_out = o3.Irreps("1e + 2e + 3x3o")
    m = Linear(irreps_in, irreps_out)
    print(m)
    m(torch.randn(irreps_in.dim))
