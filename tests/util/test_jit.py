import pytest
import warnings

import torch

from e3nn.o3 import FullyConnectedTensorProduct, Irreps
from e3nn.util.jit import script, trace_module, compile_mode
from e3nn.util.test import assert_equivariant


def test_submod_tracing():
    """Check that tracing actually occurs"""
    @compile_mode('trace')
    class BadTrace(torch.nn.Module):
        def forward(self, x):
            if x.shape[0] == 7:
                return x.new_ones(8)
            else:
                return x

        # This class has no irreps_in, so we need this to allow trace compilation
        def make_tracing_input(self):
            return {
                'forward': torch.randn(8, 3)
            }

    class ParentMod(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.child = BadTrace()

        def forward(self, x):
            return torch.as_tensor(0.5585)*self.child(x)

    parent = ParentMod()

    with pytest.raises(Exception):
        with warnings.catch_warnings():
            warnings.filterwarnings('error', category=torch.jit.TracerWarning)
            script(parent)


def test_submod_scripting():
    """Check that scripting actually occurs"""
    @compile_mode('script')
    class ScriptSubmod(torch.nn.Module):
        def forward(self, x):
            if x.shape[0] == 7:
                return x.new_zeros(8)
            else:
                return x

    class ParentMod(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.child = ScriptSubmod()

        def forward(self, x):
            return self.child(x)

    parent = ParentMod()
    assert parent(torch.randn(7, 4)).shape == (8,)

    parent_trace = trace_module(
        parent,
        inputs={
            'forward': (torch.randn(7, 4),)  # get the conditional behaviour
        }
    )
    # Does it get the behaviour it was traced for?
    assert parent_trace(torch.randn(7, 4)).shape == (8,)
    # Does it get the conditional that should have been scripted?
    x = torch.randn(5, 7)
    assert torch.allclose(parent(x), x)
    assert torch.allclose(parent_trace(x), x)


def test_compilation():
    irreps_in1 = Irreps("1e + 2e + 3x3o")
    irreps_in2 = Irreps("1e + 2e + 3x3o")
    irreps_out = Irreps("1e + 2e + 3x3o")

    # This module can't be compiled directly by TorchScript, since FullyConnectedTensorProduct is a subclass and calls super() in forward()
    class ParentModule(torch.nn.Module):
        def __init__(self, irreps_in1, irreps_in2, irreps_out):
            super().__init__()
            self.tp = FullyConnectedTensorProduct(irreps_in1, irreps_in2, irreps_out)
            self.alpha = torch.randn(1).squeeze()

        def forward(self, x1, x2):
            return self.tp(x1, x2) + self.alpha*self.tp(x1, x2)

    mod = ParentModule(irreps_in1, irreps_in2, irreps_out)
    # Try and xfail with torch.jit.script
    with pytest.raises(RuntimeError):
        mod_script = torch.jit.script(mod)
    # Compile with our compiler
    mod_script = script(mod)

    x1, x2 = irreps_in1.randn(3, -1), irreps_in2.randn(3, -1)
    assert torch.allclose(mod(x1, x2), mod_script(x1, x2))

    assert_equivariant(
        mod_script,
        irreps_in=[irreps_in1, irreps_in2],
        irreps_out=[irreps_out]
    )
