from typing import Dict

import torch
from torch import fx

from ._eval import eval_code


def _get_code(graph: fx.Graph) -> str:
    """Hack to get free function code for an fx.GraphModule"""
    code = graph.python_code('')
    return code.replace("def forward(self, ", "def main(")


class CodeGenMixin:
    """Mixin for classes that dynamically generate TorchScript code using FX.

    This class manages evaluating and compiling generated code for subclasses while remaining pickle/deepcopy compatible. If subclasses need to override ``__getstate__``/``__setstate__``, they should be sure to call CodeGenMixin's first and use its output.
    """
    def _codegen_register(
        self,
        funcs: Dict[str, fx.Graph],
        compile: bool = True
    ) -> None:
        """Register dynamically generated methods.

        Parameters
        ----------
            funcs : Dict[str, fx.Graph]
                Dictionary mapping method names to their code.
        """
        if not hasattr(self, "__codegen__"):
            # func_name -> code
            self.__codegen__ = {}
        self.__codegen__.update({
            fname: _get_code(graph)
            for fname, graph in funcs.items()
        })
        if compile:
            self._codegen_compile()

    def _codegen_compile(self):
        """Compile and set all registered dynamically generated methods."""
        if hasattr(self, "__codegen__"):
            # Compile the generated or static code
            for fname, code in self.__codegen__.items():
                setattr(
                    self,
                    fname,
                    torch.jit.script(eval_code(code)["main"])
                )

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
            # We cant save compiled functions
            for fname in self.__codegen__:
                out.pop(fname, None)
        return out

    def __setstate__(self, d):
        d = d.copy()
        if "__codegen__" in d:
            codegen_state = d.pop("__codegen__")
            # Remove any compiled methods that somehow entered the state
            for k in codegen_state:
                d.pop(k, None)
            self.__codegen__ = codegen_state
            self._codegen_compile()

        # We need to check if other parent classes of self define __getstate__
        if hasattr(super(CodeGenMixin, self), "__setstate__"):
            super(CodeGenMixin, self).__setstate__(d)
        else:
            self.__dict__.update(d)
