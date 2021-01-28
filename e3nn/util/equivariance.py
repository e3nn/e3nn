import random

import torch
from torch._C import Value

from e3nn import o3


def equivariance_error(func, irreps_in=None, irreps_out=None, ntrials=1, batch_dim=True, do_parity=True):
    r"""Get the maximum equivariance error for ``func`` over ``ntrials``

    Each trial randomizes the input parameters and the batch dimension.

    Parameters
    ----------
    func : callable
        the function to test
    irreps_in : list of `Irreps` or `Irreps`
        the input irreps for each of the arguments of ``func``. If left as the default of ``None``, ``[func.irreps_in]`` will be used.
    irreps_out : list of `Irreps` or `Irreps`
        the out irreps for each of the arguments of ``func``. If left as the default of ``None``, ``[func.irreps_out]`` will be used.
    ntrials : int
        run this many trials with random inputs
    batch_dim : bool or tuple
        if True (the default), the input to ``func`` will be given a random batch dimension. If a tuple, the tuple gives the bounds on the size of the random batch dimension.
    do_parity : True
        whether to test parity

    Returns
    -------
    `torch.Tensor`
        scalar tensor giving largest componentwise error
    """
    if irreps_in is None:
        irreps_in = [func.irreps_in]
    if irreps_out is None:
        irreps_out = [func.irreps_out]

    if isinstance(irreps_in, o3.Irreps):
        irreps_in = [irreps_in]
    elif isinstance(irreps_in, list) or isinstance(irreps_in, tuple):
        irreps_in = [o3.Irreps(i) for i in irreps_in]
    else:
        irreps_in = [o3.Irreps(irreps_in)]

    if isinstance(irreps_out, o3.Irreps):
        irreps_out = [irreps_out]
    elif isinstance(irreps_out, list) or isinstance(irreps_out, tuple):
        irreps_out = [o3.Irreps(i) for i in irreps_out]
    else:
        irreps_out = [o3.Irreps(irreps_out)]

    if isinstance(batch_dim, tuple):
        assert len(batch_dim) == 2
        assert batch_dim[0] > 0

    if do_parity:
        parity_ks = torch.Tensor([0, 1])
    else:
        parity_ks = [None]

    biggest_err = -float("Inf")
    for trial in range(ntrials):
        if batch_dim:
            # Test with random batch dimensions each trial
            if isinstance(batch_dim, tuple):
                arg_shape = (random.randint(*batch_dim), -1)
            else:
                arg_shape = (random.randint(1, 5), -1)
        else:
            arg_shape = (-1,)

        for parity_k in parity_ks:
            args = [irreps.randn(*arg_shape) for irreps in irreps_in]
            D_params = o3.rand_angles()
            # Also add parity
            D_params = D_params + (parity_k,)
            # Evaluate the function on rotated arguments:
            rot_args = [a @ irreps.D_from_angles(*D_params).T for irreps, a in zip(irreps_in, args)]
            x1 = func(*rot_args)
            print('x1', x1.shape)
            # Evaluate the function on the arguments, then rotate:
            x2 = func(*args)
            print('x2', x2.shape)
            # Deal with output shapes
            if len(irreps_out) > 1:
                assert len(x1) == len(irreps_out)
                assert len(x2) == len(irreps_out)
                # x1 is already a sequence of outputs
                x1 = list(x1)
                x2 = [a @ irreps.D_from_angles(*D_params).T for irreps, a in zip(irreps_out, x2)]
            else:
                x1 = [x1]
                x2 = [x2 @ irreps_out[0].D_from_angles(*D_params).T]

            assert len(x1) == len(x2)
            print('x1 len', len(x1))

            error = max(
                (a - b).abs().max()
                for a, b in zip(x1, x2)
            )
            if error > biggest_err:
                biggest_err = error

    return biggest_err
