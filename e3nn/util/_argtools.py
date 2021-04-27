from typing import Optional

import random
import warnings

import torch

from e3nn.o3 import Irreps


def _transform(dat, irreps_dat, rot_mat, translation=0.):
    """Transform ``dat`` by ``rot_mat`` and ``translation`` according to ``irreps_dat``."""
    out = []
    for irreps, a in zip(irreps_dat, dat):
        if irreps is None:
            out.append(a)
        elif irreps == 'cartesian_points':
            translation = torch.as_tensor(translation, device=a.device)
            out.append((a @ rot_mat.T.to(a.device)) + translation)
        else:
            # For o3.Irreps
            out.append(a @ irreps.D_from_matrix(rot_mat).T.to(a.device))
    return out


def _get_io_irreps(func, irreps_in=None, irreps_out=None):
    """Preprocess or, if not given, try to infer the I/O irreps for ``func``."""
    SPECIAL_VALS = ['cartesian_points', None]

    if (irreps_in is None or irreps_out is None) and isinstance(func, torch.jit.ScriptModule):
        warnings.warn(
            "Asking to infer irreps in/out of a compiled TorchScript module. This is unreliable, please provide `irreps_in` and `irreps_out` explicitly."
        )

    if irreps_in is None:
        if hasattr(func, 'irreps_in'):
            irreps_in = func.irreps_in  # gets checked for type later
        elif hasattr(func, 'irreps_in1'):
            irreps_in = [func.irreps_in1, func.irreps_in2]
        else:
            raise ValueError("Cannot infer irreps_in for %r; provide them explicitly" % func)
    if irreps_out is None:
        if hasattr(func, 'irreps_out'):
            irreps_out = func.irreps_out  # gets checked for type later
        else:
            raise ValueError("Cannot infer irreps_out for %r; provide them explicitly" % func)

    if isinstance(irreps_in, Irreps) or irreps_in in SPECIAL_VALS:
        irreps_in = [irreps_in]
    elif isinstance(irreps_in, list):
        irreps_in = [i if i in SPECIAL_VALS else Irreps(i) for i in irreps_in]
    else:
        if isinstance(irreps_in, tuple) and not isinstance(irreps_in, Irreps):
            warnings.warn(
                f"Module {func} had irreps_in of type tuple but not Irreps; ambiguous whether the tuple should be interpreted as a tuple representing a single Irreps or a tuple of objects each to be converted to Irreps. Assuming the former. If the latter, use a list."
            )
        irreps_in = [Irreps(irreps_in)]

    if isinstance(irreps_out, Irreps) or irreps_out in SPECIAL_VALS:
        irreps_out = [irreps_out]
    elif isinstance(irreps_out, list):
        irreps_out = [i if i in SPECIAL_VALS else Irreps(i) for i in irreps_out]
    else:
        if isinstance(irreps_in, tuple) and not isinstance(irreps_in, Irreps):
            warnings.warn(
                f"Module {func} had irreps_out of type tuple but not Irreps; ambiguous whether the tuple should be interpreted as a tuple representing a single Irreps or a tuple of objects each to be converted to Irreps. Assuming the former. If the latter, use a list."
            )
        irreps_out = [Irreps(irreps_out)]

    return irreps_in, irreps_out


def _get_args_in(func, args_in=None, irreps_in=None, irreps_out=None):
    irreps_in, irreps_out = _get_io_irreps(func, irreps_in=irreps_in, irreps_out=irreps_out)
    if args_in is None:
        args_in = _rand_args(irreps_in)
    assert len(args_in) == len(irreps_in), "irreps_in and args_in don't match in length"
    return args_in, irreps_in, irreps_out


def _rand_args(irreps_in, batch_size: Optional[int] = None):
    if not all((isinstance(i, Irreps) or i == 'cartesian_points') for i in irreps_in):
        raise ValueError("Random arguments cannot be generated when argument types besides Irreps and `'cartesian_points'` are specified; provide explicit ``args_in``")
    if batch_size is None:
        # Generate random args with random size batch dim between 1 and 4:
        batch_size = random.randint(1, 4)
    args_in = [
        torch.randn(batch_size, 3) if (irreps == 'cartesian_points') else irreps.randn(batch_size, -1)
        for irreps in irreps_in
    ]
    return args_in


def _get_device(mod: torch.nn.Module) -> torch.device:
    # Try to a get a parameter
    a_buf = next(mod.parameters(), None)
    if a_buf is None:
        # If there isn't one, try to get a buffer
        a_buf = next(mod.buffers(), None)
    return a_buf.device if a_buf is not None else 'cpu'


def _get_floating_dtype(mod: torch.nn.Module) -> torch.dtype:
    """Guess floating dtype for module.

    Assumes no mixed precision.
    """
    # Try to a get a parameter
    a_buf = None
    for buf in mod.parameters():
        if buf.is_floating_point():
            a_buf = buf
            break
    if a_buf is None:
        # If there isn't one, try to get a buffer
        for buf in mod.buffers():
            if buf.is_floating_point():
                a_buf = buf
                break
    return a_buf.dtype if a_buf is not None else torch.get_default_dtype()


def _to_device_dtype(args, device=None, dtype=None):
    kwargs = {}
    if device is not None:
        kwargs['device'] = device
    if dtype is not None:
        kwargs['dtype'] = dtype

    if isinstance(args, torch.Tensor):
        if args.is_floating_point():
            # Only convert dtypes of floating tensors
            return args.to(device=device, dtype=dtype)
        else:
            return args.to(device=device)
    elif isinstance(args, tuple):
        return tuple(_to_device_dtype(e, **kwargs) for e in args)
    elif isinstance(args, list):
        return [_to_device_dtype(e, **kwargs) for e in args]
    elif isinstance(args, dict):
        return{k: _to_device_dtype(v, **kwargs) for k, v in args.items()}
    else:
        raise TypeError("Only (nested) dict/tuple/lists of Tensors can be moved to a device/dtype.")
