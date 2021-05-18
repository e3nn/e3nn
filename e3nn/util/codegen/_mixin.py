from typing import Dict, Union, Tuple, Callable
import io

import torch
from torch import fx


def _make_autograd_func(forward: torch.jit.ScriptModule, backward: torch.jit.ScriptModule) -> Callable:
    # Make a singleton autograd function
    # TODO: cache these based on IRs?
    class _MyFunc(torch.autograd.Function):
        @staticmethod
        def forward(ctx, *args):
            ctx.save_for_backward(*args)
            return _MyFunc._forward(*args)

        @staticmethod
        def backward(ctx, *grads):
            args = ctx.saved_tensors + tuple(grads)
            return _MyFunc._backward(*args)
    _MyFunc._forward = forward
    _MyFunc._backward = backward
    return _MyFunc.apply


def _scriptmodule_to_bytes(smod: torch.jit.ScriptModule) -> bytes:
    assert isinstance(smod, torch.jit.ScriptModule)
    # Save the compiled code as TorchScript IR
    buffer = io.BytesIO()
    torch.jit.save(smod, buffer)
    # Serialize that IR (just some `bytes`) instead of
    # the ScriptModule
    return buffer.getvalue()


def _scriptmodule_from_bytes(buffer: bytes) -> torch.jit.ScriptModule:
    # Make sure bytes, not ScriptModules, got made
    assert isinstance(buffer, bytes)
    # Load the TorchScript IR buffer
    buffer = io.BytesIO(buffer)
    smod = torch.jit.load(buffer)
    assert isinstance(smod, torch.jit.ScriptModule)
    return smod


class CodeGenMixin:
    """Mixin for classes that dynamically generate TorchScript code using FX.

    This class manages evaluating and compiling generated code for subclasses
    while remaining pickle/deepcopy compatible. If subclasses need to override
    ``__getstate__``/``__setstate__``, they should be sure to call CodeGenMixin's
    implimentation first and use its output.
    """
    def _codegen_register(
        self,
        funcs: Dict[str, Union[fx.GraphModule, Tuple[fx.GraphModule, fx.GraphModule]]],
    ) -> None:
        """Register ``fx.GraphModule``s as TorchScript submodules.

        Parameters
        ----------
            funcs : Dict[str, Union[fx.GraphModule, Tuple[fx.GraphModule, fx.GraphModule]]]
                Dictionary mapping submodule names to graph modules.
                If a value is a two-tuple of graph modules, the first is the forward pass and the second is the backward pass.
        """
        if not hasattr(self, "__codegen__"):
            # list of submodule names that are managed by this object
            self.__codegen__ = {}

        for fname, graphmod in funcs.items():
            if isinstance(graphmod, fx.GraphModule):
                forward = graphmod
                backward = None
            elif isinstance(graphmod, tuple):
                assert len(graphmod) == 2
                forward, backward = graphmod
                assert isinstance(forward, fx.GraphModule)
                assert isinstance(backward, fx.GraphModule)
            else:
                raise TypeError(f"Invalid code to register `{graphmod}`")

            forward = torch.jit.script(forward)
            assert isinstance(forward, torch.jit.ScriptModule)

            if backward is None:
                self.__codegen__[fname] = forward
                # Add the ScriptModule as a submodule so it can be called
                setattr(self, fname, forward)
            else:
                backward = torch.jit.script(backward)
                assert isinstance(backward, torch.jit.ScriptModule)
                self.__codegen__[fname] = (forward, backward)
                setattr(self, fname, _make_autograd_func(forward, backward))

    # In order to support copy.deepcopy and pickling, we need to not save the compiled TorchScript functions:
    # See pickle docs: https://docs.python.org/3/library/pickle.html#pickling-class-instances
    def __getstate__(self):
        # - Get a state to work with -
        # We need to check if other parent classes of self define __getstate__
        # torch.nn.Module does not currently impliment __get/setstate__ but
        # may in the future, which is why we have these hasattr checks for
        # other superclasses.
        if hasattr(super(CodeGenMixin, self), "__getstate__"):
            out = super(CodeGenMixin, self).__getstate__()
        else:
            out = self.__dict__

        out = out.copy()
        # We need a copy of the _modules OrderedDict
        # Otherwise, modifying the returned state will modify the current module itself
        out["_modules"] = out["_modules"].copy()

        # - Add saved versions of the ScriptModules to the state -
        codegen_state = {}
        if hasattr(self, "__codegen__"):
            for fname, smod in self.__codegen__.items():
                if isinstance(smod, torch.jit.ScriptModule):
                    codegen_state[fname] = _scriptmodule_to_bytes(smod)
                    # Remove the compiled submodule from being a submodule
                    # of the saved module
                    del out["_modules"][fname]
                else:
                    codegen_state[fname] = tuple(_scriptmodule_to_bytes(mod) for mod in smod)
                    # no need to remove submodule since it was in an autograd function
                    # but since its an attribute, and not a submodule, we need to 
                    # remove it from the __dict__ instead:
                    del out[fname]

            out["__codegen__"] = codegen_state
        return out

    def __setstate__(self, d):
        d = d.copy()
        # We don't want to add this to the object when we call super's __setstate__
        codegen_state = d.pop("__codegen__", None)

        # We need to initialize self first so that we can add submodules
        # We need to check if other parent classes of self define __getstate__
        if hasattr(super(CodeGenMixin, self), "__setstate__"):
            super(CodeGenMixin, self).__setstate__(d)
        else:
            self.__dict__.update(d)

        if codegen_state is not None:
            new_codegen_state = {}
            for fname, buffer in codegen_state.items():
                assert isinstance(fname, str)
                if isinstance(buffer, bytes):
                    # its just a forward
                    # Add the ScriptModule as a submodule
                    smod = _scriptmodule_from_bytes(buffer)
                    setattr(self, fname, smod)
                    new_codegen_state[fname] = smod
                elif isinstance(buffer, tuple):
                    assert len(buffer) == 2
                    forward, backward = (_scriptmodule_from_bytes(b) for b in buffer)
                    new_codegen_state[fname] = (forward, backward)
                    setattr(self, fname, _make_autograd_func(forward, backward))
                else:
                    raise TypeError
            self.__codegen__ = new_codegen_state
