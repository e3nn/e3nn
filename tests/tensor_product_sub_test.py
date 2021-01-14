import torch
from e3nn import o3
from e3nn.nn import Identity
from e3nn.o3 import FullyConnectedTensorProduct, Linear, FullTensorProduct


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


def test_full():
    torch.set_default_dtype(torch.float64)

    irreps_in1 = o3.Irreps("1e + 2e + 3x3o")
    irreps_in2 = o3.Irreps("1e + 2x2e + 2x3o")
    m = FullTensorProduct(irreps_in1, irreps_in2)
    print(m)

    x1 = irreps_in1.randn(-1)
    x2 = irreps_in2.randn(-1)
    q = o3.rand_quaternion()

    a = m.irreps_out.D_from_quaternion(q) @ m(x1, x2)
    b = m(irreps_in1.D_from_quaternion(q) @ x1, irreps_in2.D_from_quaternion(q) @ x2)

    assert (a - b).abs().max() < 1e-10
