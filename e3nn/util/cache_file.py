'''
Cache in files
'''
import fcntl
import glob
import os
import pickle
import sys
from functools import lru_cache, wraps
from itertools import count


class FileSystemMutex:
    '''
    Mutual exclusion of different **processes** using the file system
    '''

    def __init__(self, filename):
        self.handle = None
        self.filename = filename

    def acquire(self):
        '''
        Locks the mutex
        if it is already locked, it waits (blocking function)
        '''
        self.handle = open(self.filename, 'w')
        fcntl.lockf(self.handle, fcntl.LOCK_EX)
        self.handle.write("{}\n".format(os.getpid()))
        self.handle.flush()

    def release(self):
        '''
        Unlock the mutex
        '''
        if self.handle is None:
            raise RuntimeError()
        fcntl.lockf(self.handle, fcntl.LOCK_UN)
        self.handle.close()
        self.handle = None

    def __enter__(self):
        self.acquire()

    def __exit__(self, exc_type, exc_value, traceback):
        self.release()


def cached_picklesjar(dirname, maxsize=128, open_jar=open):
    '''
    Cache a function with a directory

    :param dirname: the directory path
    :param maxsize: maximum size of the RAM cache (there is no limit for the directory cache)
    '''

    def decorator(func):
        '''
        The actual decorator
        '''

        @lru_cache(maxsize=maxsize)
        @wraps(func)
        def wrapper(*args, **kwargs):
            '''
            The wrapper of the function
            '''
            try:
                os.makedirs(dirname)
            except FileExistsError:
                pass
            except PermissionError:
                return func(*args, **kwargs)

            if not os.access(dirname, os.W_OK):
                return func(*args, **kwargs)

            mutexfile = os.path.join(dirname, "mutex")

            key = (args, frozenset(kwargs.items()), func.__defaults__)

            with FileSystemMutex(mutexfile):
                for file in glob.glob(os.path.join(dirname, "*.cache")):
                    with open_jar(file, "rb") as file:
                        loadedkey = pickle.load(file)
                        if key == loadedkey:
                            return pickle.load(file)

            print("compute... ", end="")
            sys.stdout.flush()
            result = func(*args, **kwargs)
            print("save... ", end="")
            sys.stdout.flush()

            with FileSystemMutex(mutexfile):
                for i in count():
                    file = os.path.join(dirname, "{}.cache".format(i))
                    if not os.path.isfile(file):
                        break

                try:
                    with open_jar(file, "wb") as file:
                        pickle.dump(key, file)
                        pickle.dump(result, file)
                    print("done")
                except PermissionError:
                    pass

            return result

        return wrapper

    return decorator
