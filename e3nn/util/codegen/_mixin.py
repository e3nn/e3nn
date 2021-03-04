from typing import Dict, Union

from ._eval import eval_code
from ._lazy import LazyCodeGenerator


class CodeGenMixin:
    """Mixin for classes that dynamically generate some of their methods.

    This class manages evaluating generated code for subclasses while remaining pickle/deepcopy compatible. If subclasses need to override ``__getstate__``/``__setstate__``, they should be sure to call CodeGenMixin's first and use its output.
    """
    def _codegen_register(
        self,
        funcs: Dict[str, Union[str, LazyCodeGenerator]],
        compile: bool = True
    ) -> None:
        """Register dynamically generated methods.

        Parameters
        ----------
            funcs : Dict[str, Union[str, LazyCodeGenerator]]
                Dictionary mapping method names to their code.
        """
        if not hasattr(self, "__codegen__"):
            # (code_dict, generator_dict)
            self.__codegen__ = ({}, {})
        self.__codegen__[0].update({
            k: v for k, v in funcs.items() if isinstance(v, str)
        })
        self.__codegen__[1].update({
            k: v for k, v in funcs.items() if isinstance(v, LazyCodeGenerator)
        })
        if compile:
            self._codegen_compile()

    def _codegen_compile(self):
        """Compile and set all registered dynamically generated methods.

        Reruns any ``LazyCodeGenerator``s.
        """
        if hasattr(self, "__codegen__"):
            # Run generators and update code
            self.__codegen__[0].update({
                f: g.generate() for f, g in self.__codegen__[1].items()
            })
            # Compile the generated or static code
            for fname, code in self.__codegen__[0].items():
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
            # Save only code strings, we can't save LazyCodeGenerator
            out["__codegen__"] = self.__codegen__[0]
            # Functions that have code and not just generators are those that are currently compiled, so we remove them
            for fname in self.__codegen__[0]:
                out.pop(fname, None)
        return out

    def __setstate__(self, d):
        d = d.copy()
        if "__codegen__" in d:
            codegen_state = d.pop("__codegen__")
            # Remove any compiled methods that somehow entered the state
            for k in codegen_state:
                d.pop(k, None)
            self.__codegen__ = (codegen_state, {})  # no generators
            self._codegen_compile()

        # We need to check if other parent classes of self define __getstate__
        if hasattr(super(CodeGenMixin, self), "__setstate__"):
            super(CodeGenMixin, self).__setstate__(d)
        else:
            self.__dict__.update(d)
