import warnings
import inspect

import torch

from ._argtools import _get_io_irreps, _rand_args

_E3NN_COMPILE_MODE = "__e3nn_compile_mode__"
_MAKE_TRACING_INPUTS = '_make_tracing_inputs'


def compile_mode(mode: str):
    if not mode in ['trace', 'script']:
        raise ValueError("Invalid compile mode")
    def decorator(obj):
        if not (inspect.isclass(obj) and issubclass(obj, torch.nn.Module)):
            raise TypeError("@e3nn.util.jit.compile_mode can only decorate classes derived from torch.nn.Module")
        if hasattr(obj, _E3NN_COMPILE_MODE):
            # TODO: is this right for subclasses?
            warnings.warn("Something is strange — did this class get marked twice with @e3nn.util.jit.compile_mode?")
        setattr(obj, _E3NN_COMPILE_MODE, mode)
        return obj
    return decorator


def get_compile_mode(mod: torch.nn.Module) -> str:
    if hasattr(mod, _E3NN_COMPILE_MODE):
        mode = getattr(mod, _E3NN_COMPILE_MODE)
    else:
        mode = getattr(type(mod), _E3NN_COMPILE_MODE, None)
    assert mode in ['script', 'trace', None], "Invalid compile mode `%r`" % mode
    return mode


def compile(
    mod: torch.nn.Module,
    n_trace_checks: int = 1,
    script_options: dict = {},
    trace_options: dict = {},
):
    """Recursively compile all submodules according to their decorators.

    Submodules without decorators will be unaffected.
    """
    # TODO: debug logging
    assert n_trace_checks >= 1
    # == recurse to children ==
    # This allows us to trace compile submodules of modules we are going to script
    for submod_name, submod in mod.named_children():
        setattr(
            mod,
            submod_name,
            compile(
                submod,
                n_trace_checks=n_trace_checks,
                script_options=script_options,
                trace_options=trace_options
            )
        )
    # == Compile this module now ==
    mode = get_compile_mode(mod)

    if mode == 'script':
        mod = torch.jit.script(mod, **script_options)
    elif mode == 'trace':
        # These are always modules, so we're always using trace_module
        # We need tracing inputs:
        check_inputs = get_tracing_inputs(mod, n_trace_checks)
        assert len(check_inputs) >= 1, "Must have at least one tracing input."
        # Do the actual trace
        mod = torch.jit.trace_module(
            mod,
            inputs=check_inputs[0],
            check_inputs=check_inputs,
            **trace_options
        )
    return mod


def get_tracing_inputs(mod: torch.nn.Module, n: int = 1):
    if hasattr(mod, _MAKE_TRACING_INPUTS):
        # This returns a trace_module style dict of method names to test inputs
        trace_inputs = mod._make_tracing_inputs(n)
        assert isinstance(trace_inputs, list)
        for d in trace_inputs:
            assert isinstance(d, dict), "_make_tracing_inputs must return a list of dict[str, tuple]"
            assert all(isinstance(k, str) and isinstance(v, tuple) for k, v in d.items()), "_make_tracing_inputs must return a list of dict[str, tuple]"
    else:
        # Try to infer. This will throw if it can't.
        irreps_in, _ = _get_io_irreps(
            mod,
            irreps_out=[None]  # we're only trying to infer inputs
        )
        trace_inputs = [{'forward': _rand_args(irreps_in)} for _ in range(n)]
    return trace_inputs


def trace_module(
    mod: torch.nn.Module,
    inputs: dict = None,
    check_inputs: list = [],
):
    # Set the compile mode for mod, temporarily
    old_mode = getattr(mod, _E3NN_COMPILE_MODE, None)
    if old_mode is not None and old_mode != 'trace':
        warnings.warn(f"Trying to trace a module of type {type(mod).__name__} marked with @compile_mode != 'trace', expect errors!")
    setattr(mod, _E3NN_COMPILE_MODE, 'trace')

    # If inputs are provided, set make_tracing_input temporarily
    old_make_tracing_input = None
    if inputs is not None:
        old_make_tracing_input = getattr(mod, _MAKE_TRACING_INPUTS, None)
        setattr(
            mod,
            _MAKE_TRACING_INPUTS,
            lambda num: ([inputs] + check_inputs)
        )

    # Compile
    out = compile(mod)

    # Restore old values, if we had them
    if old_mode is not None:
        setattr(mod, _E3NN_COMPILE_MODE, old_mode)
    if old_make_tracing_input is not None:
        setattr(mod, _MAKE_TRACING_INPUTS, old_make_tracing_input)
    return out


def trace(
    mod: torch.nn.Module,
    example_inputs: tuple = None,
    check_inputs: list = [],
):
    return trace_module(
        mod=mod,
        inputs=({'forward': example_inputs} if example_inputs is not None else None),
        check_inputs=[{'forward': c} for c in check_inputs],
    )


def script(mod: torch.nn.Module):
    # Set the compile mode for mod, temporarily
    old_mode = getattr(mod, _E3NN_COMPILE_MODE, None)
    if old_mode is not None and old_mode != 'script':
        warnings.warn(f"Trying to script a module of type {type(mod).__name__} marked with @compile_mode != 'script', expect errors!")
    setattr(mod, _E3NN_COMPILE_MODE, 'script')

    # Compile
    out = compile(mod)

    # Restore old values, if we had them
    if old_mode is not None:
        setattr(mod, _E3NN_COMPILE_MODE, old_mode)

    return out
