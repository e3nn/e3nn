"""
Evaluate a python string as code
"""
import importlib.machinery
import importlib.util
import tempfile
import functools
from typing import Dict


@functools.lru_cache(None)
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


class CodeGenMixin:
    """Mixin for classes that dynamically generate some of their methods.

    This class manages evaluating generated code for subclasses while remaining pickle/deepcopy compatible. If subclasses need to override ``__getstate__``/``__setstate__``, they should be sure to call CodeGenMixin's first and use its output.
    """
    def _codegen_register(self, funcs: Dict[str, str]) -> None:
        """Register dynamically generated methods.

        Parameters
        ----------
            funcs : Dict[str, str]
                Dictionary mapping method names to their code.
        """
        if not hasattr(self, "__codegen__"):
            self.__codegen__ = {}
        self.__codegen__.update(funcs)
        self._codegen_compile()

    def _codegen_compile(self):
        """Compile and set all registered dynamically generated methods."""
        if hasattr(self, "__codegen__"):
            for fname, code in self.__codegen__.items():
                setattr(self, fname, eval_code(code).main)

    # In order to support copy.deepcopy and pickling, we need to not save the compiled TorchScript functions:
    # See pickle docs: https://docs.python.org/3/library/pickle.html#pickling-class-instances
    # torch.nn.Module does not currently impliment __get/setstate__ but may in the future, which is why we have these hasattr checks for other superclasses.
    def __getstate__(self):
        # - Get a state to work with -
        # We need to check if other parent classes of self define __getstate__
        if hasattr(super(CodeGenMixin, self), "__getstate__"):
            out = super(CodeGenMixin, self).__getstate__().copy()
        else:
            out = self.__dict__.copy()
        # - Remove compiled methods -
        if hasattr(self, "__codegen__"):
            out["__codegen__"] = self.__codegen__
            for fname in self.__codegen__:
                out.pop(fname, None)
        return out

    def __setstate__(self, d):
        d = d.copy()
        if "__codegen__" in d:
            # Remove any compiled methods that somehow entered the state
            for k in d['__codegen__']:
                d.pop(k, None)
            # Set for self
            self.__dict__['__codegen__'] = d.pop("__codegen__")
            # And compile
            self._codegen_compile()

        # We need to check if other parent classes of self define __getstate__
        if hasattr(super(CodeGenMixin, self), "__setstate__"):
            super(CodeGenMixin, self).__setstate__(d)
        else:
            self.__dict__.update(d)
