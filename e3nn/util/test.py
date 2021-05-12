from typing import Iterable, Optional

import random
import math
import itertools
import warnings
import logging
import inspect

import torch

from e3nn import o3
from e3nn.util.jit import compile, get_tracing_inputs, get_compile_mode, _MAKE_TRACING_INPUTS
from ._argtools import _get_args_in, _get_io_irreps, _transform, _rand_args


# Make a logger for reporting error statistics
logger = logging.getLogger(__name__)


def _logging_name(func) -> str:
    """Get a decent string representation of ``func`` for logging"""
    if inspect.isfunction(func):
        return func.__name__
    else:
        return repr(func)


# The default float tolerance
FLOAT_TOLERANCE = {
    t: torch.as_tensor(v, dtype=t)
    for t, v in {
        torch.float32: 1e-3,
        torch.float64: 1e-10
    }.items()
}


try:
    # If pytest is available, define an e3nn pytest plugin
    # See https://docs.pytest.org/en/stable/fixture.html#using-fixtures-from-other-projects
    import pytest

    @pytest.fixture(scope='session', autouse=True, params=['float32', 'float64'])
    def float_tolerance(request):
        """Run all tests with various PyTorch default dtypes.

        This is a session-wide, autouse fixture — you only need to request it explicitly if a test needs to know the tolerance for the current default dtype.

        Returns
        --------
            A precision threshold to use for closeness tests.
        """
        old_dtype = torch.get_default_dtype()
        dtype = {
            'float32': torch.float32,
            'float64': torch.float64
        }[request.param]
        torch.set_default_dtype(dtype)
        yield FLOAT_TOLERANCE[dtype]
        torch.set_default_dtype(old_dtype)
except ImportError:
    pass


def random_irreps(
    n: int = 1,
    lmax: int = 4,
    mul_min: int = 0,
    mul_max: int = 5,
    len_min: int = 0,
    len_max: int = 4,
    clean: bool = False,
    allow_empty: bool = True
):
    r"""Generate random irreps parameters for testing.

    Parameters
    ----------
        n : int, optional
            How many to generate; defaults to 1.
        lmax : int, optional
            The maximum L to generate (inclusive); defaults to 4.
        mul_min : int, optional
            The smallest multiplicity to generate, defaults to 0.
        mul_max : int, optional
            The largest multiplicity to generate, defaults to 5.
        len_min : int, optional
            The smallest number of irreps to generate, defaults to 0.
        len_max : int, optional
            The largest number of irreps to generate, defaults to 4.
        clean : bool, optional
            If ``True``, only ``o3.Irreps`` objects will be returned. If ``False`` (the default), ``Irreps``-like objects like strings and lists of tuples can be returned.
        allow_empty : bool, optional
            Whether to allow generating empty ``Irreps``.
    Returns
    -------
        An irreps-like object if ``n == 1`` or a list of them if ``n > 1``
    """
    assert n >= 1
    assert lmax >= 0
    assert mul_min >= 0
    assert mul_max >= mul_min

    if not allow_empty and len_min == 0:
        len_min = 1
    assert len_min >= 0
    assert len_max >= len_min

    out = []
    for _ in range(n):
        this_irreps = []
        for _ in range(random.randint(len_min, len_max)):
            this_irreps.append((
                random.randint(mul_min, mul_max),
                (random.randint(0, lmax), random.choice((1, -1)))
            ))
        if not allow_empty and all(m == 0 for m, _ in this_irreps):
            this_irreps[-1] = (random.randint(1, mul_max), this_irreps[-1][1])
        this_irreps = o3.Irreps(this_irreps)

        if clean:
            outtype = "irreps"
        else:
            outtype = random.choice(("irreps", "str", "list"))
        if outtype == "irreps":
            out.append(this_irreps)
        elif outtype == "str":
            out.append(repr(this_irreps))
        elif outtype == "list":
            out.append([(mul_ir.mul, (mul_ir.ir.l, mul_ir.ir.p)) for mul_ir in this_irreps])

    if n == 1:
        return out[0]
    else:
        return out


def format_equivariance_error(errors: dict) -> str:
    """Format the dictionary returned by ``equivariance_error`` into a readable string.

    Parameters
    ----------
        errors : dict
            A dictionary of errors returned by ``equivariance_error``.

    Returns
    -------
        A string.
    """
    return "; ".join(
        "(parity_k={:d}, did_translate={}) -> error={:.3e}".format(
            int(k[0]),
            bool(k[1]),
            float(v)
        )
        for k, v in errors.items()
    )


def assert_equivariant(
    func,
    args_in=None,
    irreps_in=None,
    irreps_out=None,
    tolerance=None,
    **kwargs
) -> dict:
    r"""Assert that ``func`` is equivariant.

    Parameters
    ----------
        args_in : list or None
            the original input arguments for the function. If ``None`` and the function has ``irreps_in`` consisting only of ``o3.Irreps`` and ``'cartesian'``, random test inputs will be generated.
        irreps_in : object
            see ``equivariance_error``
        irreps_out : object
            see ``equivariance_error``
        tolerance : float or None
            the threshold below which the equivariance error must fall. If ``None``, (the default), ``FLOAT_TOLERANCE[torch.get_default_dtype()]`` is used.
        **kwargs : kwargs
            passed through to ``equivariance_error``.

    Returns
    -------
    The same as ``equivariance_error``: a dictionary mapping tuples ``(parity_k, did_translate)`` to errors
    """
    # Prevent pytest from showing this function in the traceback
    __tracebackhide__ = True

    args_in, irreps_in, irreps_out = _get_args_in(
        func,
        args_in=args_in,
        irreps_in=irreps_in,
        irreps_out=irreps_out
    )

    # Get error
    errors = equivariance_error(
        func,
        args_in=args_in,
        irreps_in=irreps_in,
        irreps_out=irreps_out,
        **kwargs
    )

    logger.info(
        "Tested equivariance of `%s` -- max componentwise errors: %s",
        _logging_name(func),
        format_equivariance_error(errors),
    )

    # Check it
    if tolerance is None:
        tolerance = FLOAT_TOLERANCE[torch.get_default_dtype()]

    problems = {case: err for case, err in errors.items() if err > tolerance}

    if len(problems) != 0:
        errstr = "Largest componentwise equivariance error was too large for: "
        errstr += format_equivariance_error(problems)
        assert len(problems) == 0, errstr

    return errors


def equivariance_error(
    func,
    args_in,
    irreps_in=None,
    irreps_out=None,
    ntrials=1,
    do_parity=True,
    do_translation=True
):
    r"""Get the maximum equivariance error for ``func`` over ``ntrials``

    Each trial randomizes the equivariant transformation tested.

    Parameters
    ----------
    func : callable
        the function to test
    args_in : list
        the original inputs to pass to ``func``.
    irreps_in : list of `Irreps` or `Irreps`
        the input irreps for each of the arguments in ``args_in``. If left as the default of ``None``, ``get_io_irreps`` will be used to try to infer them. If a sequence is provided, valid elements are also the string ``'cartesian'``, which denotes that the corresponding input should be dealt with as cartesian points in 3D, and ``None``, which indicates that the argument should not be transformed.
    irreps_out : list of `Irreps` or `Irreps`
        the out irreps for each of the return values of ``func``. Accepts similar values to ``irreps_in``.
    ntrials : int
        run this many trials with random transforms
    do_parity : bool
        whether to test parity
    do_translation : bool
        whether to test translation for ``'cartesian'`` inputs

    Returns
    -------
    dictionary mapping tuples ``(parity_k, did_translate)`` to errors
    """
    irreps_in, irreps_out = _get_io_irreps(func, irreps_in=irreps_in, irreps_out=irreps_out)

    if do_parity:
        parity_ks = torch.Tensor([0, 1])
    else:
        parity_ks = torch.Tensor([0])

    if ('cartesian_points' not in irreps_in):
        # There's nothing to translate
        do_translation = False
    if do_translation:
        do_translation = [False, True]
    else:
        do_translation = [False]

    tests = itertools.product(parity_ks, do_translation)

    neg_inf = -float("Inf")
    biggest_errs = {}

    for trial in range(ntrials):
        for this_test in tests:
            parity_k, this_do_translate = this_test
            # Build a rotation matrix for point data
            rot_mat = o3.rand_matrix()
            # add parity
            rot_mat *= (-1)**parity_k
            # build translation
            translation = 10 * torch.randn(1, 3, dtype=rot_mat.dtype) if this_do_translate else 0.

            # Evaluate the function on rotated arguments:
            rot_args = _transform(args_in, irreps_in, rot_mat, translation)
            x1 = func(*rot_args)

            # Evaluate the function on the arguments, then apply group action:
            x2 = func(*args_in)

            # Deal with output shapes
            assert type(x1) == type(x2), f"Inconsistant return types {type(x1)} and {type(x2)}"
            if isinstance(x1, torch.Tensor):
                # Make sequences
                x1 = [x1]
                x2 = [x2]
            elif isinstance(x1, list) or isinstance(x1, tuple):
                # They're already tuples
                x1 = list(x1)
                x2 = list(x2)
            else:
                raise TypeError(f"equivariance_error cannot handle output type {type(x1)}")
            assert len(x1) == len(x2)
            assert len(x1) == len(irreps_out)

            # apply the group action to x2
            x2 = _transform(x2, irreps_out, rot_mat, translation)

            error = max(
                (a - b).abs().max()
                for a, b in zip(x1, x2)
            )

            if error > biggest_errs.get(this_test, neg_inf):
                biggest_errs[this_test] = error

    return biggest_errs


# TODO: this is only for things marked with @compile_mode.
# Make something else for general script/traceability
def assert_auto_jitable(
    func,
    error_on_warnings=True,
    n_trace_checks=2,
    strict_shapes=True,
):
    r"""Assert that submodule ``func`` is automatically JITable.

    Parameters
    ----------
        func : Callable
            The function to trace.
        error_on_warnings : bool
            If True (default), TracerWarnings emitted by ``torch.jit.trace`` will be treated as errors.
        n_random_tests : int
            If ``args_in`` is ``None`` and arguments are being automatically generated, this many random arguments will be generated as test inputs for ``torch.jit.trace``.
        strict_shapes : bool
            Test that the traced function errors on inputs with feature dimensions that don't match the input irreps.
    Returns
    -------
        The traced TorchScript function.
    """
    # Prevent pytest from showing this function in the traceback
    __tracebackhide__ = True

    if get_compile_mode(func) is None:
        raise ValueError("assert_auto_jitable is only for modules marked with @compile_mode")

    # Test tracing
    with warnings.catch_warnings():
        if error_on_warnings:
            warnings.filterwarnings('error', category=torch.jit.TracerWarning)
        func_jit = compile(
            func,
            n_trace_checks=n_trace_checks
        )

    # Confirm that it rejects incorrect shapes
    # This check only makes sense if all inputs are Tensors with irreps; otherwise we can't know how to modify the arguments or that our modifications make them wrong.
    if strict_shapes and not hasattr(func, _MAKE_TRACING_INPUTS):
        try:
            all_bad_args = get_tracing_inputs(func, n=1)[0]
        except ValueError:
            # couldn't infer, don't check
            pass
        else:
            for method, bad_args in all_bad_args.items():
                # Since _rand_args is OK, they're all Irreps style args where changing the feature dimension is wrong
                bad_which = random.randint(0, len(bad_args) - 1)
                bad_args = list(bad_args)
                bad_args[bad_which] = bad_args[bad_which][..., :-random.randint(1, 3)]  # make bad shape
                try:
                    if method == 'forward':
                        func_jit(*bad_args)
                    else:
                        getattr(func_jit, method)(*bad_args)
                except (torch.jit.Error, RuntimeError):
                    # As far as I can tell, there's no good way to introspect TorchScript exceptions.
                    pass
                else:
                    raise AssertionError("Traced function didn't error on bad input shape")

    return func_jit


# TODO: custom in_vars, out_vars support
def assert_normalized(
    func: torch.nn.Module,
    irreps_in=None,
    irreps_out=None,
    normalization: str = "component",
    n_input: int = 10_000,
    n_weight: Optional[int] = None,
    weights: Optional[Iterable[torch.nn.Parameter]] = None,
    atol: float = 0.1,
) -> None:
    r"""Assert that ``func`` is normalized.

    See https://docs.e3nn.org/en/stable/guide/normalization.html for more information on the normalization scheme.

    ``atol``, ``n_input``, and ``n_weight`` may need to be significantly higher in order to converge the statistics to pass the test.

    Parameters
    ----------
        func : torch.nn.Module
            the module to test
        irreps_in : object
            see ``equivariance_error``
        irreps_out : object
            see ``equivariance_error``
        normalization : str, default "component"
            one of "component" or "norm". Note that this is defined for both the inputs and the outputs; if you need seperate normalizations for input and output please file a feature request.
        n_input : int, default 10_000
            the number of input samples to use for each weight init
        n_weight : int, default 20
            the number of weight initializations to sample
        weights : optional iterable of parameters
            the weights to reinitialize ``n_weight`` times. If ``None`` (default), ``func.parameters()`` will be used.
        atol : float, default 0.1
            tolerance for checking moments. Higher values for this prevent explosive computational costs for this test.
    """
    # Prevent pytest from showing this function in the traceback
    __tracebackhide__ = True

    if normalization not in ("component", "norm"):
        raise ValueError(f"invalid normalization `{normalization}`")

    irreps_in, irreps_out = _get_io_irreps(func, irreps_in=irreps_in, irreps_out=irreps_out)

    if all(i.num_irreps == 0 for i in irreps_in) or all(i.num_irreps == 0 for i in irreps_out):
        # Short-circut
        return

    if weights is None:
        if isinstance(func, torch.nn.Module):
            weights = func.parameters()
        else:
            weights = []
    weights = list(weights)

    if len(weights) == 0:
        assert n_weight is None or n_weight == 1, "Without weights to re-init, n_weight must be 1 or None"
        n_weight = 1
    else:
        n_weight = 20 if n_weight is None else n_weight

    with torch.no_grad():
        expected_squares = [torch.zeros(irreps.dim) for irreps in irreps_out]
        n_samples = 0
        for weight_init in range(n_weight):
            # generate weight sample
            for param in weights:
                param.normal_()

            # generate input sample
            args_in = _rand_args(irreps_in, batch_size=n_input)
            # args_in gives component normalized irreps
            if normalization == "norm":
                for i, irreps in enumerate(irreps_in):
                    for mul_ir, ir_slice in zip(irreps, irreps.slices()):
                        args_in[i][:, ir_slice].div_(math.sqrt(mul_ir.ir.dim))

            # run func
            this_outs = func(*args_in)
            if not isinstance(this_outs, list) or isinstance(this_outs, tuple):
                this_outs = (this_outs,)
            assert len(this_outs) == len(irreps_out)

            # square
            this_outs = [e.square() for e in this_outs]

            # update running average
            for i in range(len(irreps_out)):
                assert this_outs[i].shape[0] == n_input
                update = this_outs[i].sum(dim=0) - n_input * expected_squares[i]
                update.div_(n_input + n_samples)
                expected_squares[i].add_(update)
            n_samples += n_input

    # check them
    for expected_square, irreps in zip(expected_squares, irreps_out):
        if irreps == "cartesian_points" or irreps is None:
            continue
        if normalization == "component":
            targets = [1.0] * len(irreps)
        elif normalization == "norm":
            targets = [1.0 / math.sqrt(ir.dim) for _, ir in irreps]

        for i, (target, ir_slice) in enumerate(zip(targets, irreps.slices())):
            if ir_slice.start == ir_slice.stop:
                continue
            max_componentwise = (expected_square[ir_slice] - target).abs().max().item()
            logger.info(
                "Tested normalization of %r: max componentwise error %.6f",
                _logging_name(func),
                max_componentwise
            )
            assert max_componentwise <= atol, f"< x_i^2 > !~= {target:.6f} for output irrep #{i}, {irreps[i]}. Max componentwise error: {max_componentwise:.6f}"
