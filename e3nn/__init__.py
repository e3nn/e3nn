__version__ = "0.3.2"


from typing import Dict

_OPT_DEFAULTS: Dict[str, bool] = dict(
    specialized_code=True,
    optimize_einsums=True
)


def set_optimization_defaults(
    specialized_code: bool = True,
    optimize_einsums: bool = True
) -> None:
    r"""Globally set the default optimization settings.

    Parameters
    ----------
    specialized_code : bool, default True
        Whether to use specialized code for (combinations of) irreps for which it exists.
    optimize_einsums : bool, default True
        Whether to use ``opt_einsum_fx``.
    """
    _OPT_DEFAULTS['specialized_code'] = specialized_code
    _OPT_DEFAULTS['optimize_einsums'] = optimize_einsums


def get_optimization_defaults() -> Dict[str, bool]:
    r"""Get the global default optimization settings."""
    return dict(_OPT_DEFAULTS)
