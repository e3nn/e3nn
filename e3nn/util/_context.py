from abc import ABC, abstractmethod
from contextlib import AbstractContextManager
from functools import wraps


class AbstractContextDecoratorManager(ABC, AbstractContextManager):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def __enter__(self):
        pass

    @abstractmethod
    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def __call__(self, f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            with self:
                return f(*args, **kwargs)
        return wrapper
