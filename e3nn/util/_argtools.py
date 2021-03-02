import random

import torch

from e3nn.o3 import Irreps


def _transform(dat, irreps_dat, rot_mat, translation=0.):
    """Transform ``dat`` by ``rot_mat`` and ``translation`` according to ``irreps_dat``."""
    out = []
    for irreps, a in zip(irreps_dat, dat):
        if irreps is None:
            out.append(a)
        elif irreps == 'cartesian_points':
            out.append((a @ rot_mat.T) + translation)
        else:
            # For o3.Irreps
            out.append(a @ irreps.D_from_matrix(rot_mat).T)
    return out


def _get_io_irreps(func, irreps_in=None, irreps_out=None):
    """Preprocess or, if not given, try to infer the I/O irreps for ``func``."""
    SPECIAL_VALS = ['cartesian_points', None]

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

    if isinstance(irreps_in, Irreps) or irreps_in in SPECIAL_VALS:
        irreps_in = [irreps_in]
    elif isinstance(irreps_in, list) or isinstance(irreps_in, tuple):
        irreps_in = [i if i in SPECIAL_VALS else Irreps(i) for i in irreps_in]
    else:
        irreps_in = [Irreps(irreps_in)]

    if isinstance(irreps_out, Irreps) or irreps_out in SPECIAL_VALS:
        irreps_out = [irreps_out]
    elif isinstance(irreps_out, list) or isinstance(irreps_out, tuple):
        irreps_out = [i if i in SPECIAL_VALS else Irreps(i) for i in irreps_out]
    else:
        irreps_out = [Irreps(irreps_out)]

    return irreps_in, irreps_out


def _get_args_in(func, args_in=None, irreps_in=None, irreps_out=None):
    irreps_in, irreps_out = _get_io_irreps(func, irreps_in=irreps_in, irreps_out=irreps_out)
    if args_in is None:
        args_in = _rand_args(irreps_in)
    assert len(args_in) == len(irreps_in), "irreps_in and args_in don't match in length"
    return args_in, irreps_in, irreps_out


def _rand_args(irreps_in):
    if not all((isinstance(i, Irreps) or i == 'cartesian_points') for i in irreps_in):
        raise ValueError("Random arguments cannot be generated when argument types besides Irreps and `'cartesian_points'` are specified; provide explicit ``args_in``")
    # Generate random args with random size batch dim between 1 and 4:
    batch_size = random.randint(1, 4)
    args_in = [
        torch.randn(batch_size, 3) if (irreps == 'cartesian_points') else irreps.randn(batch_size, -1)
        for irreps in irreps_in
    ]
    return args_in


def _to_device(args, device):
    if isinstance(args, torch.Tensor):
        return args.to(device=device)
    elif isinstance(args, tuple):
        return tuple(_to_device(e, device) for e in args)
    elif isinstance(args, list):
        return [_to_device(e, device) for e in args]
    elif isinstance(args, dict):
        return{k: _to_device(v, device) for k, v in args.items()}
    else:
        raise TypeError("Only (nested) dict/tuple/lists of Tensors can be moved to a device.")
