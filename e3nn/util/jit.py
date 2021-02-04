import warnings
import inspect

import torch

from ._argtools import _get_io_irreps, _rand_args

_E3NN_COMPILE_MODE = "__e3nn_compile_mode__"
_MAKE_TRACING_INPUT = 'make_tracing_input'


def compile_mode(mode: str):
    if not mode in ['trace', 'script']:
        raise ValueError("Invalid compile mode")
    def decorator(obj):
        if not (inspect.isclass(obj) and issubclass(obj, torch.nn.Module)):
            raise TypeError("@e3nn.util.jit.script_if_tracing can only decorate classes derived from torch.nn.Module")
        if hasattr(obj, _E3NN_COMPILE_MODE):
            warnings.warn("Something is strange — did this class get marked twice with @e3nn.util.jit.script_if_tracing?")
        setattr(obj, _E3NN_COMPILE_MODE, mode)
        return obj
    return decorator


def _compile_submodules(
    mod: torch.nn.Module,
    n_extra_trace_checks: int = 0,
    script_options: dict = {},
    trace_options: dict = {},
):
    """Recursively compile all submodules according to their decorators.

    Submodules without decorators will be unaffected.
    """
    # == recurse to children ==
    # This allows us to trace compile submodules of modules we are going to script
    for submod_name, submod in mod.named_children():
        setattr(
            mod,
            submod_name,
            _compile_submodules(
                submod,
                n_extra_trace_checks=n_extra_trace_checks,
                script_options=script_options,
                trace_options=trace_options
            )
        )
    # == Compile this module now ==
    if hasattr(mod, _E3NN_COMPILE_MODE):
        compile_mode = getattr(mod, _E3NN_COMPILE_MODE)
    else:
        compile_mode = getattr(type(mod), _E3NN_COMPILE_MODE, None)
    assert compile_mode in ['script', 'trace', None]

    if compile_mode == 'script':
        mod = torch.jit.script(mod, **script_options)
    elif compile_mode == 'trace':
        # These are always modules, so we're always using trace_module
        # We need tracing inputs:
        if hasattr(mod, _MAKE_TRACING_INPUT):
            # This returns a trace_module style dict of method names to test inputs
            check_inputs = [mod.make_tracing_input() for _ in range(n_extra_trace_checks+1)]
            assert all(isinstance(e, dict) for e in check_inputs), "make_tracing_input must return a dict"
        else:
            # Try to infer. This will throw if it can't.
            irreps_in, _ = _get_io_irreps(
                mod,
                irreps_out=[None]  # we're only trying to infer inputs
            )
            check_inputs = [{'forward': _rand_args(irreps_in)} for _ in range(n_extra_trace_checks+1)]
        # Do the actual trace
        mod = torch.jit.trace_module(
            mod,
            inputs=check_inputs[0],
            check_inputs=check_inputs,
            **trace_options
        )
    return mod


def trace_module(
    mod: torch.nn.Module,
    inputs: dict = None,
    n_extra_trace_checks: int = 0,
    script_options: dict = {},
    trace_options: dict = {}
):
    # Set the compile mode for mod, temporarily
    old_mode = getattr(mod, _E3NN_COMPILE_MODE, None)
    setattr(mod, _E3NN_COMPILE_MODE, 'trace')

    # If inputs are provided, set make_tracing_input temporarily
    old_make_tracing_input = None
    if inputs is not None:
        old_make_tracing_input = getattr(mod, _MAKE_TRACING_INPUT, None)
        setattr(
            mod,
            _MAKE_TRACING_INPUT,
            lambda: inputs
        )

    # Compile
    out = _compile_submodules(
        mod,
        n_extra_trace_checks=n_extra_trace_checks,
        script_options=script_options,
        trace_options=trace_options
    )

    # Restore old values, if we had them
    if old_mode is not None:
        setattr(mod, _E3NN_COMPILE_MODE, old_mode)
    if old_make_tracing_input is not None:
        setattr(mod, _MAKE_TRACING_INPUT, old_make_tracing_input)
    return out


def script(
    mod: torch.nn.Module,
    n_extra_trace_checks: int = 0,
    script_options: dict = {},
    trace_options: dict = {}
):
    # Set the compile mode for mod, temporarily
    old_mode = getattr(mod, _E3NN_COMPILE_MODE, None)
    setattr(mod, _E3NN_COMPILE_MODE, 'script')

    # Compile
    out = _compile_submodules(
        mod,
        n_extra_trace_checks=n_extra_trace_checks,
        script_options=script_options,
        trace_options=trace_options
    )

    # Restore old values, if we had them
    if old_mode is not None:
        setattr(mod, _E3NN_COMPILE_MODE, old_mode)

    return out
