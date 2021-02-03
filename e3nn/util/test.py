import random
import itertools
import warnings

import torch

from e3nn import o3


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


def assert_equivariant(
    func,
    args_in=None,
    irreps_in=None,
    irreps_out=None,
    tolerance=None,
    **kwargs
):
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
    # Check it
    if tolerance is None:
        tolerance = FLOAT_TOLERANCE[torch.get_default_dtype()]

    problems = {case: err for case, err in errors.items() if err > tolerance}

    if len(problems) != 0:
        print(problems)
        errstr = (
            "Largest componentwise equivariance error was too large for: " + \
            '; '.join("(parity_k={:d}, did_translate={}) -> error={:.3e}".format(int(k[0]), bool(k[1]), float(v)) for k, v in problems.items())
        )
        assert len(problems) == 0, errstr


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
            translation = 10*torch.randn(1, 3, dtype=rot_mat.dtype) if this_do_translate else 0.

            # Evaluate the function on rotated arguments:
            rot_args = _transform(args_in, irreps_in, rot_mat, translation)
            x1 = func(*rot_args)

            # Evaluate the function on the arguments, then apply group action:
            x2 = func(*args_in)

            # Deal with output shapes
            if len(irreps_out) == 1:
                # Make sequences
                x1 = [x1]
                x2 = [x2]
            else:
                # They're already tuples
                x1 = list(x1)
                x2 = list(x2)
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


def assert_jit_trace(
    func,
    method_name=None,
    args_in=None,
    irreps_in=None,
    irreps_out=None,
    n_random_tests=2,
    strict_shapes=True,
    **kwargs
):
    r"""Assert that ``func`` can be traced to TorchScript.

    Parameters
    ----------
        func : Callable
            The function to trace.
        method_name : str or None (default)
            If ``func`` is a module, methods other than ``forward()`` can be traced by giving their name as a string. (This uses ``torch.jit.trace_module`` instead of ``torch.jit.trace``.)
        args_in : list or None
            the original input arguments for the function. If ``None`` and the function has ``irreps_in`` consisting only of ``o3.Irreps`` and ``'cartesian'``, random test inputs will be generated.
        irreps_in : object
            see ``equivariance_error``
        irreps_out : object
            see ``equivariance_error``
        n_random_tests : int
            If ``args_in`` is ``None`` and arguments are being automatically generated, this many random arguments will be generated as test inputs for ``torch.jit.trace``.
        strict_shapes : bool
            Test that the traced function errors on inputs with feature dimensions that don't match the input irreps.
        **kwargs : kwargs
            passed through to ``torch.jit.trace``.
    Returns
    -------
        The traced TorchScript function.
    """
    # Prevent pytest from showing this function in the traceback
    __tracebackhide__ = True

    random_tests = args_in is None

    args_in, irreps_in, irreps_out = _get_args_in(
        func,
        args_in=args_in,
        irreps_in=irreps_in,
        irreps_out=irreps_out
    )

    if random_tests:
        test_inputs = [args_in] + [_rand_args(irreps_in) for _ in range(n_random_tests)]
    else:
        test_inputs = [args_in]

    # Test tracing
    with warnings.catch_warnings():
        warnings.filterwarnings('error', category=torch.jit.TracerWarning)
        if method_name is not None:
            func_trace = torch.jit.trace_module(
                func,
                inputs={method_name: tuple(args_in)},
                check_inputs=[{method_name: t} for t in test_inputs]
            )
            func_trace = getattr(func_trace, method_name)
        else:
            func_trace = torch.jit.trace(
                func,
                example_inputs=tuple(args_in),
                check_inputs=test_inputs
            )

    # Confirm that it rejects incorrect shapes
    if random_tests and strict_shapes:
        bad_args = _rand_args(irreps_in)
        # Since _rand_args is OK, they're all Irreps style args where changing the feature dimension is wrong
        bad_which = random.randint(0, len(bad_args)-1)
        bad_args[bad_which] = bad_args[bad_which][..., :-random.randint(1, 3)]  # make bad shape
        try:
            func_trace(*bad_args)
        except torch.jit.Error as e:
            # As far as I can tell, there's no good way to introspect TorchScript exceptions. Checking for RuntimeError at least eliminates particlar possibilities
            assert "RuntimeError" in str(e), "TorchScript through an unexpectedly strange error"
        else:
            raise AssertionError("Traced function didn't error on bad input shape")

    return func_trace


def _transform(dat, irreps_dat, rot_mat, translation=0.):
    """Transform ``dat`` by ``rot_mat`` and ``translation`` according to ``irreps_dat``."""
    out = []
    for irreps, a in zip(irreps_dat, dat):
        if irreps is None:
            out.append(a)
        elif irreps == 'cartesian_points':
            out.append((a @ rot_mat.T) + translation)
        elif irreps == 'cartesian_vectors':
            out.append((a @ rot_mat.T))
        else:
            # For o3.Irreps
            out.append(a @ irreps.D_from_matrix(rot_mat).T)
    return out


def _get_io_irreps(func, irreps_in=None, irreps_out=None):
    """Preprocess or, if not given, try to infer the I/O irreps for ``func``."""
    SPECIAL_VALS = ['cartesian_points', 'cartesian_vectors', None]

    if irreps_in is None:
        if hasattr(func, 'irreps_in'):
            irreps_in = [func.irreps_in]
        elif hasattr(func, 'irreps_in1'):
            irreps_in = [func.irreps_in1, func.irreps_in2]
        else:
            raise ValueError("Cannot infer irreps_in for %r; provide them explicitly" % func)
    if irreps_out is None:
        if hasattr(func, 'irreps_out'):
            irreps_out = [func.irreps_out]
        else:
            raise ValueError("Cannot infer irreps_out for %r; provide them explicitly" % func)

    if isinstance(irreps_in, o3.Irreps) or irreps_in in SPECIAL_VALS:
        irreps_in = [irreps_in]
    elif isinstance(irreps_in, list) or isinstance(irreps_in, tuple):
        irreps_in = [i if i in SPECIAL_VALS else o3.Irreps(i) for i in irreps_in]
    else:
        irreps_in = [o3.Irreps(irreps_in)]

    if isinstance(irreps_out, o3.Irreps) or irreps_out in SPECIAL_VALS:
        irreps_out = [irreps_out]
    elif isinstance(irreps_out, list) or isinstance(irreps_out, tuple):
        irreps_out = [i if i in SPECIAL_VALS else o3.Irreps(i) for i in irreps_out]
    else:
        irreps_out = [o3.Irreps(irreps_out)]

    return irreps_in, irreps_out


def _get_args_in(func, args_in=None, irreps_in=None, irreps_out=None):
    irreps_in, irreps_out = _get_io_irreps(func, irreps_in=irreps_in, irreps_out=irreps_out)
    if args_in is None:
        args_in = _rand_args(irreps_in)
    assert len(args_in) == len(irreps_in), "irreps_in and args_in don't match in length"
    return args_in, irreps_in, irreps_out


def _rand_args(irreps_in):
    if not all((isinstance(i, o3.Irreps) or i == 'cartesian_points') for i in irreps_in):
        raise ValueError("Random arguments cannot be generated when argument types besides Irreps and `'cartesian_points'` are specified; provide explicit ``args_in``")
    # Generate random args with random size batch dim between 1 and 4:
    batch_size = random.randint(1, 4)
    args_in = [
        torch.randn(batch_size, 3) if (irreps == 'cartesian_points') else irreps.randn(batch_size, -1)
        for irreps in irreps_in
    ]
    return args_in
