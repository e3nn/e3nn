# Please see PR #203 for these commented out code: https://github.com/e3nn/e3nn/pull/203

# from abc import ABCMeta, abstractmethod
# from contextlib import AbstractContextManager
# from functools import wraps


# class AbstractContextDecoratorManager(AbstractContextManager, metaclass=ABCMeta):
#     def __init__(self):
#         super().__init__()

#     @abstractmethod
#     def __enter__(self):
#         pass

#     @abstractmethod
#     def __exit__(self, exc_type, exc_value, traceback):
#         pass

#     def __call__(self, f):
#         @wraps(f)
#         def wrapper(*args, **kwargs):
#             with self:
#                 return f(*args, **kwargs)
#         return wrapper
