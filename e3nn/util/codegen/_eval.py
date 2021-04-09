"""
Evaluate a python string as code
"""
import importlib.machinery
import importlib.util
import tempfile
import functools
import os.path


# Set a large but finite maximum size to prevent long-running or unusual client codes from growing memory use without bound.
@functools.lru_cache(maxsize=512)
def eval_code(code):
    r"""
    save code in a temporary file and import it as a module
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        new_filename = os.path.join(temp_dir, "__gencode.py")
        with open(new_filename, 'w+b') as new_file:
            new_file.write(bytes(code, 'ascii'))
        loader = importlib.machinery.SourceFileLoader('main', new_filename)
        spec = importlib.util.spec_from_loader(loader.name, loader)
        mod = importlib.util.module_from_spec(spec)
        loader.exec_module(mod)
    return mod
