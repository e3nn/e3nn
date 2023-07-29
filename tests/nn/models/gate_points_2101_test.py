import copy
import random
import tempfile

import pytest
import torch

from e3nn import o3
from e3nn.nn.models.gate_points_2101 import Network
from e3nn.util.test import assert_auto_jitable, assert_equivariant


@pytest.fixture
def network():
    num_nodes = 5
    irreps_in = o3.Irreps("3x0e + 2x1o")
    irreps_attr = o3.Irreps("10x0e")
    irreps_out = o3.Irreps("2x0o + 2x1o + 2x2e")

    f = Network(
        irreps_in,
        o3.Irreps("5x0e + 5x0o + 5x1e + 5x1o"),
        irreps_out,
        irreps_attr,
        o3.Irreps.spherical_harmonics(3),
        layers=3,
        max_radius=2.0,
        number_of_basis=5,
        radial_layers=2,
        radial_neurons=100,
        num_neighbors=4.0,
        num_nodes=num_nodes,
    )

    def random_graph():
        N = random.randint(3, 7)
        return {"pos": torch.randn(N, 3), "x": f.irreps_in.randn(N, -1), "z": f.irreps_node_attr.randn(N, -1)}

    return f, random_graph


def test_convolution_jit(network) -> None:
    f, _ = network
    # Get a convolution from the network
    assert_auto_jitable(f.layers[0].first)


def test_gate_points_2101_equivariant(network) -> None:
    f, random_graph = network

    # -- Test equivariance: --
    def wrapper(pos, x, z):
        data = dict(pos=pos, x=x, z=z, batch=torch.zeros(pos.shape[0], dtype=torch.long))
        return f(data)

    assert_equivariant(
        wrapper,
        irreps_in=["cartesian_points", f.irreps_in, f.irreps_node_attr],
        irreps_out=[f.irreps_out],
    )


def test_copy(network) -> None:
    f, random_graph = network
    fcopy = copy.deepcopy(f)
    g = random_graph()
    assert torch.allclose(f(g), fcopy(g))


def test_save(network) -> None:
    f, random_graph = network
    # Get a saved, loaded network
    with tempfile.NamedTemporaryFile(suffix=".pth") as tmp:
        torch.save(f, tmp.name)
        f2 = torch.load(tmp.name)
    x = random_graph()
    assert torch.all(f(x) == f2(x))
    # Get a double-saved network
    with tempfile.NamedTemporaryFile(suffix=".pth") as tmp:
        torch.save(f2, tmp.name)
        f3 = torch.load(tmp.name)
    assert torch.all(f(x) == f3(x))
