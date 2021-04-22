"""
Evaluate a python string as code
"""
import functools
import inspect

from torch import fx


# Set a large but finite maximum size to prevent long-running or unusual client codes from growing memory use without bound.
@functools.lru_cache(maxsize=512)
def eval_code(code):
    r"""
    Evaluate ``code`` and return its globals as a dict.

    Uses ``torch.fx`` and some hacks to maintain TorchScript compatability.
    """
    globals_tmp = {}
    fx.graph_module.exec_with_source(code, globals_tmp)
    if "main" in globals_tmp and inspect.isfunction(globals_tmp["main"]):
        # TorchScript gets upset when this doesn't exist:
        globals_tmp["main"].__module__ = "e3nn.util.codegen._fake_namespace"
    return globals_tmp
