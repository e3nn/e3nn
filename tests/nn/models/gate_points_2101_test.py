import pytest

import warnings
import random
import sys
import tempfile
import subprocess
import copy

import torch
from torch_geometric.data import Data

from e3nn import o3
from e3nn.nn.models.gate_points_2101 import Network
from e3nn.util.test import assert_equivariant, assert_auto_jitable
from e3nn.util.jit import trace


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
        return {
            'pos': torch.randn(N, 3),
            'x': f.irreps_in.randn(N, -1),
            'z': f.irreps_node_attr.randn(N, -1)
        }

    return f, random_graph


def test_convolution_jit(network):
    f, _ = network
    # Get a convolution from the network
    assert_auto_jitable(f.layers[0].first)


def test_gate_points_2101_equivariant(network):
    f, random_graph = network
    # -- Test equivariance: --
    def wrapper(pos, x, z):
        data = Data(pos=pos, x=x, z=z, batch=torch.zeros(pos.shape[0], dtype=torch.long))
        return f(data)

    assert_equivariant(
        wrapper,
        irreps_in=['cartesian_points', f.irreps_in, f.irreps_node_attr],
        irreps_out=[f.irreps_out],
    )


def test_gate_points_2101_jit(network):
    f, random_graph = network

    with warnings.catch_warnings():
        warnings.filterwarnings('error', category=torch.jit.TracerWarning)
        f_traced = trace(
            f,
            example_inputs=(random_graph(),),
            # Check the compute graph on 3 other random inputs
            check_inputs=[(random_graph(),) for _ in range(3)]
        )

    dat = random_graph()
    assert torch.allclose(f(dat), f_traced(dat))

    # - Try saving, loading in another process, and running -
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save stuff
        f_traced.save(tmpdir + "/model.pt")
        torch.save(dat, tmpdir + '/dat.pt')
        # Load in new process
        with open(tmpdir + '/code.py', 'x') as code:
            code.write(f"""
import torch
# Needed for the TorchScript kernels for scatter and radius_graph
import torch_scatter
import torch_cluster
f = torch.jit.load('{tmpdir}/model.pt')
d = torch.load('{tmpdir}/dat.pt')
out = f(d)
torch.save(out, '{tmpdir}/out.pt')
""")
        # Run
        # sys.executable gives the path to the current python interpreter
        proc_res = subprocess.run([sys.executable, tmpdir + '/code.py'])
        proc_res.check_returncode()
        # Check
        out = torch.load(tmpdir + '/out.pt')
        assert torch.allclose(f(dat), out)


def test_copy(network):
    f, random_graph = network
    fcopy = copy.deepcopy(f)
    g = random_graph()
    assert torch.allclose(f(g), fcopy(g))


def test_save(network):
    f, random_graph = network
    with tempfile.NamedTemporaryFile(suffix=".pth") as tmp:
        torch.save(f, tmp.name)
        f2 = torch.load(tmp.name)
    x = random_graph()
    assert torch.all(f(x) == f2(x))