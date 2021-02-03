import warnings
import inspect

import torch

_E3NN_SCRIPT_IF_TRACING = "__e3nn_script_if_tracing__"


def script_if_tracing(obj):
    if inspect.isclass(obj):
        if not issubclass(obj, torch.nn.Module):
            raise TypeError("@e3nn.util.jit.script_if_tracing can only decorate classes derived from torch.nn.Module")
        if hasattr(obj, _E3NN_SCRIPT_IF_TRACING):
            warnings.warn("Something is strange — did this class get marked twice with @e3nn.util.jit.script_if_tracing?")
        setattr(obj, _E3NN_SCRIPT_IF_TRACING, True)
        return obj
    else:
        return torch.jit._script_if_tracing(obj)


def _compile_submodules(mod: torch.nn.Module, script_options: dict = {}):
    """Recursively compile all submodules according to their decorators.

    Submodules without decorators will be unaffected.
    """
    if getattr(mod, _E3NN_SCRIPT_IF_TRACING, False):
        # torch.jit.script already recurses
        return torch.jit.script(mod, **script_options)
    else:
        # recurse to children
        for submod_name, submod in mod.named_children():
            setattr(
                mod,
                submod_name,
                _compile_submodules(submod, script_options)
            )
        return mod


def trace(mod: torch.nn.Module, *args, **kwargs):
    mod = _compile_submodules(mod)
    return torch.jit.trace(mod, *args, **kwargs)


def trace_module(mod: torch.nn.Module, *args, **kwargs):
    mod = _compile_submodules(mod)
    return torch.jit.trace_module(mod, *args, **kwargs)


def script(mod: torch.nn.Module, *args, **kwargs):
    return torch.jit.script(mod, *args, **kwargs)
