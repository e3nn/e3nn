import copy
import inspect
import warnings
import re
from typing import Optional, Callable, Tuple
from contextlib import contextmanager
from functools import wraps

# Borrowed from https://github.com/ACEsuit/mace/blob/126f490a56020c40940f8c7dbabbb23eb58e17a1/mace/tools/compile.py#L56

try:
    import torch._dynamo as dynamo
except ImportError:
    dynamo = None
from e3nn import get_optimization_defaults, set_optimization_defaults
import torch
from torch import autograd, nn
from torch import fx
from torch.fx import symbolic_trace
from opt_einsum_fx import jitable

ModuleFactory = Callable[..., nn.Module]
TypeTuple = Tuple[type, ...]


_E3NN_COMPILE_MODE = "__e3nn_compile_mode__"
_VALID_MODES = ("trace", "script", "unsupported", None)
_MAKE_TRACING_INPUTS = "_make_tracing_inputs"


def compile_mode(mode: str):
    """Decorator to set the compile mode of a module.

    Parameters
    ----------
        mode : str
            'script', 'trace', or None
    """
    if mode not in _VALID_MODES:
        raise ValueError("Invalid compile mode")

    def decorator(obj):
        if not (inspect.isclass(obj) and issubclass(obj, torch.nn.Module)):
            raise TypeError("@e3nn.util.jit.compile_mode can only decorate classes derived from torch.nn.Module")
        setattr(obj, _E3NN_COMPILE_MODE, mode)
        return obj

    return decorator


def get_compile_mode(mod: torch.nn.Module) -> str:
    """Get the compilation mode of a module.

    Parameters
    ----------
        mod : torch.nn.Module

    Returns
    -------
    'script', 'trace', or None if the module was not decorated with @compile_mode
    """
    if hasattr(mod, _E3NN_COMPILE_MODE):
        mode = getattr(mod, _E3NN_COMPILE_MODE)
    else:
        mode = getattr(type(mod), _E3NN_COMPILE_MODE, None)
    if mode is None and isinstance(mod, fx.GraphModule):
        mode = "script"
    assert mode in _VALID_MODES, "Invalid compile mode `%r`" % mode
    return mode


def compile(
    mod: torch.nn.Module,
    n_trace_checks: int = 1,
    script_options: dict = None,
    trace_options: dict = None,
    in_place: bool = True,
    recurse: bool = True,
):
    """Recursively compile a module and all submodules according to their decorators.

    (Sub)modules without decorators will be unaffected.

    Parameters
    ----------
        mod : torch.nn.Module
            The module to compile. The module will have its submodules compiled replaced in-place.
        n_trace_checks : int, default = 1
            How many random example inputs to generate when tracing a module. Must be at least one in order to have a tracing
            input. Extra example inputs will be pased to ``torch.jit.trace`` to confirm that the traced copmute graph doesn't
            change.
        script_options : dict, default = {}
            Extra kwargs for ``torch.jit.script``.
        trace_options : dict, default = {}
            Extra kwargs for ``torch.jit.trace``.
        in_place : bool, default True
            Whether to insert the recursively compiled submodules in-place, or do a deepcopy first.
        recurse : bool, default True
            Whether to recurse through the module's children before passing the parent to TorchScript

    Returns
    -------
    Returns the compiled module.
    """
    script_options = script_options or {}
    trace_options = trace_options or {}

    mode = get_compile_mode(mod)
    if mode == "unsupported":
        raise NotImplementedError(f"{type(mod).__name__} does not support TorchScript compilation")

    if not in_place:
        mod = copy.deepcopy(mod)
    # TODO: debug logging
    assert n_trace_checks >= 1

    if recurse:
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
                    trace_options=trace_options,
                    in_place=True,  # since we deepcopied the module above, we can do inplace
                    recurse=recurse,  # always true in this branch
                ),
            )

    # == Compile this module now ==
    if mode == "script":
        if isinstance(mod, fx.GraphModule):
            mod = jitable(mod)
            # In recent PyTorch versions (probably >1.12, definitely >=2.0), PyTorch's implementation of fx.GraphModule
            # causes a warning to be raised when fx.GraphModules are compiled to TorchScript with `torch.jit.script`:
            #
            #   torch/jit/_check.py:177: UserWarning: The TorchScript type system doesn't support instance-level
            #   annotations on empty non-base types in `__init__`. Instead, either 1) use a type annotation in the
            #   class body, or 2) wrap the type in `torch.jit.Attribute`.
            #
            # Using the debugger traces this back to the following line in PyTorch:
            # https://github.com/pytorch/pytorch/blob/v2.3.1/torch/fx/graph_module.py#L446
            # Because the metadata stored by GraphModule is not relevant to the compiled TorchScript module
            # we need, it should be safe to ignore this warning. The below code suppresses this warning as
            # narrowly as possible to ensure it can still be raised from user code.
            # See also: https://github.com/pytorch/pytorch/issues/89064
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    # warnings treats this argument as a regex, but we want to match a string literal exactly, so escape it:
                    message=re.escape(
                        "The TorchScript type system doesn't support instance-level annotations on empty non-base types "
                        "in `__init__`. Instead, either 1) use a type annotation in the class body, or 2) wrap the type "
                        "in `torch.jit.Attribute`."
                    ),
                    # Being specific is good form, even though matching the message should be enough:
                    category=UserWarning,
                    module="torch",
                )
                mod = torch.jit.script(mod, **script_options)
        else:
            mod = torch.jit.script(mod, **script_options)
    elif mode == "trace":
        # These are always modules, so we're always using trace_module
        # We need tracing inputs:
        check_inputs = get_tracing_inputs(
            mod,
            n_trace_checks,
        )
        assert len(check_inputs) >= 1, "Must have at least one tracing input."
        # Do the actual trace
        mod = torch.jit.trace_module(mod, inputs=check_inputs[0], check_inputs=check_inputs, **trace_options)
    return mod


def get_tracing_inputs(
    mod: torch.nn.Module, n: int = 1, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None
):
    """Get random tracing inputs for ``mod``.

    First checks if ``mod`` has a ``_make_tracing_inputs`` method. If so, calls it with ``n`` as the single argument and
    returns its results.

    Otherwise, attempts to infer the input signature of the module using ``e3nn.util._argtools._get_io_irreps``.

    Parameters
    ----------
        mod : torch.nn.Module
        n : int, default = 1
            A hint for how many inputs are wanted. Usually n will be returned, but modules don't necessarily have to.
        device : torch.device
            The device to do tracing on. If `None` (default), will be guessed.
        dtype : torch.dtype
            The dtype to trace with. If `None` (default), will be guessed.

    Returns
    -------
    list of dict
        Tracing inputs in the format of ``torch.jit.trace_module``: dicts mapping method names like ``'forward'`` to tuples of
        arguments.
    """
    # Avoid circular imports
    from ._argtools import _get_device, _get_floating_dtype, _get_io_irreps, _rand_args, _to_device_dtype

    # - Get inputs -
    if hasattr(mod, _MAKE_TRACING_INPUTS):
        # This returns a trace_module style dict of method names to test inputs
        trace_inputs = mod._make_tracing_inputs(n)
        assert isinstance(trace_inputs, list)
        for d in trace_inputs:
            assert isinstance(d, dict), "_make_tracing_inputs must return a list of dict[str, tuple]"
            assert all(
                isinstance(k, str) and isinstance(v, tuple) for k, v in d.items()
            ), "_make_tracing_inputs must return a list of dict[str, tuple]"
    else:
        # Try to infer. This will throw if it can't.
        irreps_in, _ = _get_io_irreps(mod, irreps_out=[None])  # we're only trying to infer inputs
        trace_inputs = [{"forward": _rand_args(irreps_in)} for _ in range(n)]
    # - Put them on the right device -
    if device is None:
        device = _get_device(mod)
    if dtype is None:
        dtype = _get_floating_dtype(mod)
    # Move them
    trace_inputs = _to_device_dtype(trace_inputs, device, dtype)
    return trace_inputs


def trace_module(mod: torch.nn.Module, inputs: dict = None, check_inputs: list = None, in_place: bool = True):
    """Trace a module.

    Identical signature to ``torch.jit.trace_module``, but first recursively compiles ``mod`` using ``compile``.

    Parameters
    ----------
        mod : torch.nn.Module
        inputs : dict
        check_inputs : list of dict
    Returns
    -------
    Traced module.
    """
    check_inputs = check_inputs or []

    # Set the compile mode for mod, temporarily
    old_mode = getattr(mod, _E3NN_COMPILE_MODE, None)
    if old_mode is not None and old_mode != "trace":
        warnings.warn(
            f"Trying to trace a module of type {type(mod).__name__} marked with @compile_mode != 'trace', expect errors!"
        )
    setattr(mod, _E3NN_COMPILE_MODE, "trace")

    # If inputs are provided, set make_tracing_input temporarily
    old_make_tracing_input = None
    if inputs is not None:
        old_make_tracing_input = getattr(mod, _MAKE_TRACING_INPUTS, None)
        setattr(mod, _MAKE_TRACING_INPUTS, lambda num: ([inputs] + check_inputs))

    # Compile
    out = compile(mod, in_place=in_place)

    # Restore old values, if we had them
    if old_mode is not None:
        setattr(mod, _E3NN_COMPILE_MODE, old_mode)
    if old_make_tracing_input is not None:
        setattr(mod, _MAKE_TRACING_INPUTS, old_make_tracing_input)
    return out


def trace(mod: torch.nn.Module, example_inputs: tuple = None, check_inputs: list = None, in_place: bool = True):
    """Trace a module.

    Identical signature to ``torch.jit.trace``, but first recursively compiles ``mod`` using :func:``compile``.

    Parameters
    ----------
        mod : torch.nn.Module
        example_inputs : tuple
        check_inputs : list of tuple
    Returns
    -------
    Traced module.
    """
    check_inputs = check_inputs or []

    return trace_module(
        mod=mod,
        inputs=({"forward": example_inputs} if example_inputs is not None else None),
        check_inputs=[{"forward": c} for c in check_inputs],
        in_place=in_place,
    )


def script(mod: torch.nn.Module, in_place: bool = True):
    """Script a module.

    Like ``torch.jit.script``, but first recursively compiles ``mod`` using :func:``compile``.

    Parameters
    ----------
        mod : torch.nn.Module
    Returns
    -------
    Scripted module.
    """
    # Set the compile mode for mod, temporarily
    old_mode = getattr(mod, _E3NN_COMPILE_MODE, None)
    if old_mode is not None and old_mode != "script":
        warnings.warn(
            f"Trying to script a module of type {type(mod).__name__} marked with @compile_mode != 'script', expect errors!"
        )
    setattr(mod, _E3NN_COMPILE_MODE, "script")

    # Compile
    out = compile(mod, in_place=in_place)

    # Restore old values, if we had them
    if old_mode is not None:
        setattr(mod, _E3NN_COMPILE_MODE, old_mode)

    return out


@contextmanager
def disable_e3nn_codegen():
    """Context manager that disables the legacy PyTorch code generation used in e3nn."""
    init_val = get_optimization_defaults()["jit_script_fx"]
    set_optimization_defaults(jit_script_fx=False)
    yield
    set_optimization_defaults(jit_script_fx=init_val)


def prepare(func: ModuleFactory, allow_autograd: bool = True) -> ModuleFactory:
    """Function transform that prepares a e3nn module for torch.compile

    Args:
        func (ModuleFactory): A function that creates an nn.Module
        allow_autograd (bool, optional): Force inductor compiler to inline call to
            `torch.autograd.grad`. Defaults to True.

    Returns:
        ModuleFactory: Decorated function that creates a torch.compile compatible module
    """
    if allow_autograd:
        dynamo.allow_in_graph(autograd.grad)
    elif dynamo.allowed_functions.is_allowed(autograd.grad):
        dynamo.disallow_in_graph(autograd.grad)

    @wraps(func)
    def wrapper(*args, **kwargs):
        with disable_e3nn_codegen():
            model = func(*args, **kwargs)

        model = simplify(model)
        return model

    return wrapper


_SIMPLIFY_REGISTRY = set()


def simplify_if_compile(module: nn.Module) -> nn.Module:
    """Decorator to register a module for symbolic simplification

    The decorated module will be simplifed using `torch.fx.symbolic_trace`.
    This constrains the module to not have any dynamic control flow, see:

    https://pytorch.org/docs/stable/fx.html#limitations-of-symbolic-tracing

    Args:
        module (nn.Module): the module to register

    Returns:
        nn.Module: registered module
    """
    _SIMPLIFY_REGISTRY.add(module)
    return module


def simplify(module: nn.Module) -> nn.Module:
    """Recursively searches for registered modules to simplify with
    `torch.fx.symbolic_trace` to support compiling with the PyTorch Dynamo compiler.

    Modules are registered with the `simplify_if_compile` decorator and

    Args:
        module (nn.Module): the module to simplify

    Returns:
        nn.Module: the simplified module
    """
    simplify_types = tuple(_SIMPLIFY_REGISTRY)

    for name, child in module.named_children():
        if isinstance(child, simplify_types):
            traced = symbolic_trace(child)
            setattr(module, name, traced)
        else:
            simplify(child)

    return module
