'''
Cache in file
'''
from functools import wraps
import pickle
import gzip

def cached(filename):
    '''
    Cache a function with a file
    '''
    def decorator(func):
        '''
        The actual decorator
        '''
        @wraps(func)
        def wrapper(*args):
            '''
            The wrapper of the function
            '''
            try:
                with gzip.open(filename, "rb") as file:
                    cache = pickle.load(file)
            except FileNotFoundError:
                cache = {}

            try:
                return cache[args]
            except KeyError:
                cache[args] = result = func(*args)
                with gzip.open(filename, "wb") as file:
                    pickle.dump(cache, file)
                return result
        return wrapper
    return decorator
