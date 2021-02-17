import torch

from functools import wraps
from ._context import AbstractContextDecoratorManager


class torch_default_tensor_type(AbstractContextDecoratorManager):
    def __init__(self, dtype, device):
        super().__init__()
        self.saved_ttype = None
        self.dtype = dtype
        self.device = device

    def __enter__(self):
        if self.dtype is not None or self.device is not None:
            self.saved_ttype = torch.empty(0).type()
            torch.set_default_tensor_type(self.ttype)

    def __exit__(self, exc_type, exc_value, traceback):
        if self.saved_ttype is not None:
            torch.set_default_tensor_type(self.saved_ttype)
            self.saved_ttype = None

    @property
    def ttype(self):
        return torch.empty(0, dtype=self.dtype, device=self.device).type()


class torch_default_dtype(AbstractContextDecoratorManager):
    def __init__(self, dtype):
        super().__init__()
        self.saved_dtype = None
        self.dtype = dtype

    def __enter__(self):
        if self.dtype is not None:
            self.saved_dtype = torch.get_default_dtype()
            torch.set_default_dtype(self.dtype)

    def __exit__(self, exc_type, exc_value, traceback):
        if self.saved_dtype is not None:
            torch.set_default_dtype(self.saved_dtype)
            self.saved_dtype = None


class torch_default_device(torch_default_tensor_type):
    def __init__(self, device):
        super().__init__(None, device)


def add_type_kwargs(f):
    @wraps(f)
    def wrapper(*args, dtype=None, device=None, **kwargs):
        with torch_default_tensor_type(dtype, device):
            return f(*args, **kwargs)

    if wrapper.__doc__ is not None:
        if not wrapper.__doc__.endswith("\n"):
            wrapper.__doc__ += "\n"
        wrapper.__doc__ += r"""
- dtype and device keyword arguments will be passed to torch_default_tensor_type()
"""
    return wrapper
