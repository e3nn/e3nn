__version__ = "0.5.9"


from typing import Dict


_OPT_DEFAULTS: Dict[str, bool] = dict(specialized_code=True, optimize_einsums=True, jit_script_fx=True, jit_mode="script")


def _handle_jit_script_fx_legacy(jit_script_fx: bool, current_jit_mode: str) -> str:
    """Handle the legacy jit_script_fx flag mapping to jit_mode.

    Parameters
    ----------
    jit_script_fx : bool
        The legacy jit_script_fx flag value
    current_jit_mode : str
        The current jit_mode value

    Returns
    -------
    str
        The new jit_mode value based on the legacy mapping rules
    """
    if not jit_script_fx and current_jit_mode == "eager":
        # Keep it eager
        return "eager"
    elif not jit_script_fx:
        # Map False to eager if not already eager
        return "eager"
    elif jit_script_fx and current_jit_mode not in ["script", "inductor"]:
        # Map True to script only if not already script or inductor
        return "script"
    # In all other cases, keep current jit_mode
    return current_jit_mode


def _validate_and_set_jit_mode(jit_mode: str) -> None:
    """Validate and set the jit_mode in _OPT_DEFAULTS."""
    assert jit_mode in [
        "script",
        "inductor",
        "eager",
    ], f"Invalid jit_mode: {jit_mode}. Expected 'script', 'inductor', or 'eager'."
    _OPT_DEFAULTS["jit_mode"] = jit_mode


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

        # Handles the legacy mapping for jit_script_fx
        # to jit_mode so that old code can still work
        # with the new defaults.
        if k == "jit_script_fx":
            # Update jit_mode based on the legacy mapping
            new_jit_mode = _handle_jit_script_fx_legacy(v, _OPT_DEFAULTS["jit_mode"])
            _validate_and_set_jit_mode(new_jit_mode)
            _OPT_DEFAULTS[k] = v
        elif k == "jit_mode":
            # Validate and set the new jit_mode
            _validate_and_set_jit_mode(v)
        else:
            _OPT_DEFAULTS[k] = v


def get_optimization_defaults() -> Dict[str, bool]:
    r"""Get the global default optimization settings."""
    return dict(_OPT_DEFAULTS)
