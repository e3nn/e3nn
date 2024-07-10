__version__ = "0.5.1"


from typing import Dict


_OPT_DEFAULTS: Dict[str, bool] = dict(
    specialized_code=True,
    optimize_einsums=True,
    jit_script_fx=True,
)


def set_optimization_defaults(**kwargs) -> None:
    r"""Globally set the default optimization settings.

    Parameters
    ----------
    **kwargs
        Keyword arguments to set the default optimization settings.
    """
    for k, v in kwargs.items():
        if k not in _OPT_DEFAULTS:
            raise ValueError(f"Unknown optimization option: {k}")
        _OPT_DEFAULTS[k] = v


def get_optimization_defaults() -> Dict[str, bool]:
    r"""Get the global default optimization settings."""
    return dict(_OPT_DEFAULTS)


from e3nn import o3 as o3  # noqa: F401, E402
from e3nn import nn as nn  # noqa: F401, E402
from e3nn import io as io  # noqa: F401, E402
from e3nn import experimental as experimental  # noqa: F401, E402
