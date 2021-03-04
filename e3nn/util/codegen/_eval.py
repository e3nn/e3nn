"""
Evaluate a python string as code
"""
import importlib.machinery
import importlib.util
import tempfile
import functools


# Set a large but finite maximum size to prevent long-running or unusual client codes from growing memory use without bound.
@functools.lru_cache(maxsize=512)
def eval_code(code):
    r"""
    save code in a temporary file and import it as a module
    """
    with tempfile.NamedTemporaryFile() as new_file:
        new_file.write(bytes(code, 'ascii'))
        new_file.flush()
        loader = importlib.machinery.SourceFileLoader('main', new_file.name)
        spec = importlib.util.spec_from_loader(loader.name, loader)
        mod = importlib.util.module_from_spec(spec)
        loader.exec_module(mod)
    return mod
