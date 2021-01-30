import random

import torch

from e3nn import o3


EQUIVARIANCE_TOLERANCE = {
    t: torch.as_tensor(v, dtype=t)
    for t, v in {
        torch.float32: 1e-4,
        torch.float64: 1e-10
    }.items()
}


def assert_equivariant(
    func,
    args_in=None,
    irreps_in=None,
    irreps_out=None,
    sqrt_tolerance=False,
    tolerance_multiplier=1.,
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
        sqrt_tolerance : bool
            whether to replace the tolerance with ``sqrt(tolerance)``. Defaults to False.
        tolerance_multiplier : float
            ``tolerance`` is replaced by ``tolerance_multiplier*tolerance``. Defaults to 1.
        **kwargs : kwargs
            passed through to ``equivariance_error``.
    """
    irreps_in, irreps_out = get_io_irreps(func, irreps_in=irreps_in, irreps_out=irreps_out)
    if args_in is None:
        if not all(isinstance(i, o3.Irreps) or i == 'cartesian' for i in irreps_in):
            raise ValueError("Random arguments cannot be generated when argument types besides Irreps and `'cartesian'` are specified; provide explicit ``args_in``")
        # Generate random args with random size batch dim between 1 and 4:
        batch_size = random.randint(1, 4)
        args_in = [
            torch.randn(batch_size, 3) if irreps == 'cartesian' else irreps.randn(batch_size, -1)
            for irreps in irreps_in
        ]

    # Get error
    error = equivariance_error(
        func,
        args_in=args_in,
        irreps_in=irreps_in,
        irreps_out=irreps_out,
        **kwargs
    )
    # Check it
    tol = tolerance_multiplier*EQUIVARIANCE_TOLERANCE[torch.get_default_dtype()]
    if sqrt_tolerance:
        tol = torch.sqrt(tol)
    assert error <= tol, "Largest componentwise equivariance error %f too large" % (error,)


def equivariance_error(
    func,
    args_in,
    irreps_in=None,
    irreps_out=None,
    ntrials=1,
    do_parity=True
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
    do_parity : True
        whether to test parity

    Returns
    -------
    `torch.Tensor`
        scalar tensor giving largest componentwise error
    """
    irreps_in, irreps_out = get_io_irreps(func, irreps_in=irreps_in, irreps_out=irreps_out)

    assert len(args_in) == len(irreps_in), "irreps_in and args_in don't match in length"

    if do_parity:
        parity_ks = torch.Tensor([0, 1])
    else:
        parity_ks = torch.Tensor([0])

    biggest_err = -float("Inf")

    for trial in range(ntrials):
        for parity_k in parity_ks:
            # Build a rotation matrix for point data
            rot_mat = o3.rand_matrix()
            # add parity
            rot_mat *= (-1)**parity_k

            # Evaluate the function on rotated arguments:
            rot_args = [
                a if irreps is None else (
                    (a @ rot_mat.T) if irreps == 'cartesian' else (
                        a @ irreps.D_from_matrix(rot_mat).T
                    )
                )
                for irreps, a in zip(irreps_in, args_in)
            ]
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
            x2 = [
                a if irreps is None else (
                    (a @ rot_mat.T) if irreps == 'cartesian' else (
                        a @ irreps.D_from_matrix(rot_mat).T
                    )
                )
                for irreps, a in zip(irreps_out, x2)
            ]

            error = max(
                (a - b).abs().max()
                for a, b in zip(x1, x2)
            )

            if error > biggest_err:
                biggest_err = error

    return biggest_err


def get_io_irreps(func, irreps_in=None, irreps_out=None):
    """Preprocess or, if not given, try to infer the I/O irreps for ``func``."""
    SPECIAL_VALS = ['cartesian', None]

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
