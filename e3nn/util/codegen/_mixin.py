from typing import Dict
import io

import torch
from torch import fx


def _dummy_getstate():
    return None


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
            # list of submodule names that are managed by this object
            self.__codegen__ = []
        self.__codegen__.extend(funcs.keys())

        for fname, graph in funcs.items():
            assert isinstance(graph, fx.Graph)
            scriptmod = torch.jit.script(fx.GraphModule(
                root=self,
                graph=graph,
                class_name=fname
            ))
            assert isinstance(scriptmod, torch.jit.ScriptModule)
            # To prevent pickle from erroring, even though we don't actually try
            # to serialize the scriptmodule.
            scriptmod.__getstate__ = _dummy_getstate
            # Add the ScriptModule as a submodule so it can be called
            setattr(self, fname, scriptmod)

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
                smod = getattr(self, fname)
                # Save the compiled code as TorchScript IR
                buffer = io.BytesIO()
                torch.jit.save(smod, buffer)
                # Serialize that IR (just some `bytes`) instead of
                # the ScriptModule
                out[fname] = buffer.getvalue()
        return out

    def __setstate__(self, d):
        d = d.copy()

        # We need to initialize self first so that we can add submodules
        # We need to check if other parent classes of self define __getstate__
        if hasattr(super(CodeGenMixin, self), "__setstate__"):
            super(CodeGenMixin, self).__setstate__(d)
        else:
            self.__dict__.update(d)

        if "__codegen__" in d:
            codegen_state = d.pop("__codegen__")
            for fname in codegen_state:
                # Make sure bytes, not ScriptModules, got made
                assert isinstance(getattr(self, fname), bytes)
                # Load the TorchScript IR buffer
                buffer = d[fname]
                assert isinstance(buffer, bytes)
                buffer = io.BytesIO(buffer)
                smod = torch.jit.load(buffer)
                assert isinstance(smod, torch.jit.ScriptModule)
                # Add the ScriptModule as a submodule
                setattr(self, fname, smod)
            self.__codegen__ = codegen_state
