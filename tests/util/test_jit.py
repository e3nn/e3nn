import pytest
import warnings

import torch

from e3nn.o3 import Linear, Irreps
from e3nn.nn import FullyConnectedNet
from e3nn.util.jit import script, trace_module, compile_mode, compile
from e3nn.util.test import assert_equivariant, assert_auto_jitable


def test_submod_tracing():
    """Check that tracing actually occurs"""

    @compile_mode("trace")
    class BadTrace(torch.nn.Module):
        def forward(self, x):
            if x.shape[0] == 7:
                return x.new_ones(8)
            else:
                return x

        # This class has no irreps_in, so we need this to allow trace compilation
        def make_tracing_input(self):
            return {"forward": torch.randn(8, 3)}

    class ParentMod(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.child = BadTrace()

        def forward(self, x):
            return torch.as_tensor(0.5585) * self.child(x)

    parent = ParentMod()

    with pytest.raises(Exception):
        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=torch.jit.TracerWarning)
            script(parent)


def test_submod_scripting():
    """Check that scripting actually occurs"""

    @compile_mode("script")
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

    parent_trace = trace_module(parent, inputs={"forward": (torch.randn(7, 4),)})  # get the conditional behaviour
    # Does it get the behaviour it was traced for?
    assert parent_trace(torch.randn(7, 4)).shape == (8,)
    # Does it get the conditional that should have been scripted?
    x = torch.randn(5, 7)
    assert torch.allclose(parent(x), x)
    assert torch.allclose(parent_trace(x), x)


def test_compilation():
    class Supermod(torch.nn.Module):
        def forward(self, x):
            return x * 2.0

    @compile_mode("trace")
    class ChildMod(Supermod):
        def forward(self, x):
            return super().forward(x) * 3.0

        def _make_tracing_inputs(self, n: int):
            return [{"forward": (torch.randn(2, 3),)} for _ in range(n)]

    # This module can't be compiled directly by TorchScript, since ChildMod is a subclass and calls super() in forward()
    class ContainerMod(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.submod = ChildMod()
            self.alpha = torch.randn(1).squeeze()

        def forward(self, x):
            return self.submod(x) + self.alpha * self.submod(x)

    mod = ContainerMod()
    # Try and xfail with torch.jit.script
    with pytest.raises((RuntimeError, torch.jit.Error)):
        mod_script = torch.jit.script(mod)
    # Compile with our compiler
    mod_script = script(mod)

    x = torch.randn(3, 2)
    assert torch.allclose(mod(x), mod_script(x))


def test_equivariant():
    # Confirm that a compiled tensorproduct is still equivariant
    irreps_in = Irreps("1e + 2e + 3x3o")
    irreps_out = Irreps("1e + 2e + 3x3o")
    mod = Linear(irreps_in, irreps_out)
    mod_script = compile(mod)
    assert_equivariant(
        mod_script,
        # we provide explicit irreps because infering on a script module is not reliable
        irreps_in=irreps_in,
        irreps_out=irreps_out,
    )


def test_unsupported():
    @compile_mode("unsupported")
    class ChildMod(torch.nn.Module):
        pass

    class Supermod(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.child = ChildMod()

    mod = Supermod()
    with pytest.raises(NotImplementedError):
        mod = script(mod)


def test_trace_dtypes():
    # FullyConnectedNet is traced
    fc = FullyConnectedNet([8, 16, 8])
    # compile in a dtype other than the default
    target_dtype = {torch.float32: torch.float64, torch.float64: torch.float32}[torch.get_default_dtype()]
    fc = fc.to(dtype=target_dtype)
    for weight in fc.parameters():
        assert weight.dtype == target_dtype
    assert_auto_jitable(fc)
