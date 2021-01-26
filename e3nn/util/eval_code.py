"""
Evaluate a python string as code
"""
import importlib.machinery
import importlib.util
import os
import tempfile
import functools


@functools.lru_cache(None)
def eval_code(code):
    r"""
    save code in a temporary file and import it as a module
    """
    new_file, filename = tempfile.mkstemp(text=True)

    os.write(new_file, bytes(code, 'ascii'))
    os.close(new_file)

    loader = importlib.machinery.SourceFileLoader('main', filename)
    spec = importlib.util.spec_from_loader(loader.name, loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod
