from .default_type import (
    torch_get_default_tensor_type,
    torch_get_default_device,
    explicit_default_types,
)


def prod(x):
    """Compute the product of a sequence."""
    out = 1
    for a in x:
        out *= a
    return out


__all__ = [
    "torch_get_default_tensor_type",
    "torch_get_default_device",
    "explicit_default_types",
    "prod",
]
