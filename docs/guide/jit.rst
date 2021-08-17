=======================
TorchScript JIT Support
=======================

PyTorch provides two ways to compile code into TorchScript: `tracing and scripting <https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html>`_. Tracing follows the tensor operations on an example input, allowing complex Python control flow if that control flow does not depend on the data itself. Scripting compiles a subset of Python directly into TorchScript, allowing data-dependent control flow but only limited Python features.

This is a problem for e3nn, where many modules --- such as `e3nn.o3.TensorProduct` --- use significant Python control flow based on ``e3nn.o3.Irreps`` as well as features like inheritance that are incompatible with scripting. Other modules like ``e3nn.nn.Gate``, however, contain important but simple data-dependent control flow. Thus ``e3nn.nn.Gate`` needs to be scripted, even though it contains a `e3nn.o3.TensorProduct` that has to be traced.

To hide this complexity from the user and prevent difficult-to-understand errors, ``e3nn`` implements a wrapper for ``torch.jit`` --- `e3nn.util.jit <../api/util/jit.rst>`_ --- that recursively and automatically compiles submodules according to directions they provide. Using the ``@compile_mode`` decorator, modules can indicate whether they should be scripted, traced, or left alone.

Simple Example: Scripting
=========================

We define a simple module that includes data-dependent control flow:

.. jupyter-execute::

    import torch
    from e3nn.o3 import Norm, Irreps

    class MyModule(torch.nn.Module):
        def __init__(self, irreps_in):
            super().__init__()
            self.norm = Norm(irreps_in)

        def forward(self, x):
            norm = self.norm(x)
            if torch.any(norm > 7.):
                return norm
            else:
                return norm * 0.5

    irreps = Irreps("2x0e + 1x1o")
    mod = MyModule(irreps)

To compile it to TorchScript, we can try to use ``torch.jit.script``:

.. jupyter-execute::

    try:
        mod_script = torch.jit.script(mod)
    except:
        print("Compilation failed!")

This fails because ``Norm`` is a subclass of `e3nn.o3.TensorProduct` and TorchScript doesn't support inheritance. If we use ``e3nn.util.jit.script``, on the other hand, it works:

.. jupyter-execute::

    from e3nn.util.jit import script, trace
    mod_script = script(mod)

Internally, ``e3nn.util.jit.script`` recurses through the submodules of ``mod``, compiling each in accordance with its ``@e3nn.util.jit.compile_mode`` decorator if it has one. In particular, ``Norm`` and other `e3nn.o3.TensorProduct` s are marked with ``@compile_mode('trace')``, so ``e3nn.util.jit`` constructs an example input for ``mod.norm``, traces it, and replaces it with the traced TorchScript module. Then when the parent module ``mod`` is compiled inside ``e3nn.util.jit.script`` with ``torch.jit.script``, the submodule ``mod.norm`` has already been compiled and is integrated without issue.

As expected, the scripted module and the original give the same results:

.. jupyter-execute::

    x = irreps.randn(2, -1)
    assert torch.allclose(mod(x), mod_script(x))

Mixing Tracing and Scripting
============================

Say we define:

.. jupyter-execute::

    from e3nn.util.jit import compile_mode

    @compile_mode('script')
    class MyModule(torch.nn.Module):
        def __init__(self, irreps_in):
            super().__init__()
            self.norm = Norm(irreps_in)

        def forward(self, x):
            norm = self.norm(x)
            for row in norm:
                if torch.any(row > 0.1):
                    return row
            return norm

    class AnotherModule(torch.nn.Module):
        def __init__(self, irreps_in):
            super().__init__()
            self.mymod = MyModule(irreps_in)

        def forward(self, x):
            return self.mymod(x) + 3.

And trace an instance of ``AnotherModule`` using `e3nn.util.jit.trace`:

.. jupyter-execute::

    mod2 = AnotherModule(irreps)
    example_inputs = (irreps.randn(3, -1),)
    mod2_traced = trace(
        mod2,
        example_inputs
    )

Note that we marked ``MyModule`` with ``@compile_mode('script')`` because it contains control flow, and that the control flow is preserved even when called from the traced ``AnotherModule``:

.. jupyter-execute::

    print(mod2_traced(torch.zeros(2, irreps.dim)))
    print(mod2_traced(irreps.randn(3, -1)))

We can confirm that the submodule ``mymod`` was compiled as a script, but that ``mod2`` was traced:

.. jupyter-execute::

    print(type(mod2_traced))
    print(type(mod2_traced.mymod))

Customizing Tracing Inputs
==========================

Submodules can also be compiled automatically using tracing if they are marked with ``@compile_mode('trace')``. When submodules are compiled by tracing it must be possible to generate plausible input examples on the fly.

These example inputs can be generated automatically based on the ``irreps_in`` of the module (the specifics are the same as for ``assert_equivariant``). If this is not possible or would yield incorrect results, a module can define a ``_make_tracing_inputs`` method that generates example inputs of correct shape and type.

.. jupyter-execute::

    @compile_mode('trace')
    class TracingModule(torch.nn.Module):
        def forward(self, x: torch.Tensor, indexes: torch.LongTensor):
            return x[indexes].sum()

        # Because this module has no `irreps_in`, and because
        # `irreps_in` can't describe indexes, since it's a LongTensor,
        # we impliment _make_tracing_inputs
        def _make_tracing_inputs(self, n: int):
            import random
            # The compiler asks for n example inputs ---
            # this is only a suggestion, the only requirement
            # is that at least one be returned.
            return [
                {
                    'forward': (
                        torch.randn(5, random.randint(1, 3)),
                        torch.arange(3)
                    )
                }
                for _ in range(n)
            ]

To recursively compile this module and its submodules in accordance with their ``@compile_mode``s, we can use ``e3nn.util.jit.compile`` directly. This can be useful if the module you are compiling is annotated with ``@compile_mode`` and you don't want to override that annotation by using ``trace`` or ``script``:

.. jupyter-execute::

    from e3nn.util.jit import compile
    mod3 = TracingModule()
    mod3_traced = compile(mod3)
    print(type(mod3_traced))

Deciding between ``'script'`` and ``'trace'``
=============================================

The easiest way to decide on a compile mode for your module is to try both. Tracing will usually generate warnings if it encounters dynamic control flow that it cannot fully capture, and scripting will raise compiler errors for features it does not support.

In general, any module that uses inheritance or control flow based on ``e3nn.o3.Irreps`` in ``forward()`` will have to be traced.

Testing
=======

A helper function is provided to unit test that auto-JITable modules (those annotated with ``@compile_mode``) can be compiled:

.. jupyter-execute::

    from e3nn.util.test import assert_auto_jitable
    assert_auto_jitable(mod2)

By default, ``assert_auto_jitable`` will test traced modules to confirm that they reject input shapes that are likely incorrect. Specifically, it changes ``x.shape[-1]`` on the assumption that the final dimension is a network architecture constant. If this heuristic is wrong for your module (like it is for ``TracedModule`` above), it can be disabled:

.. jupyter-execute::

    assert_auto_jitable(mod3, strict_shapes=False)

Compile mode ``"unsupported"``
==============================

Sometimes you may write modules that use features unsupported by TorchScript regardless of whether you trace or script. To avoid cryptic errors from TorchScript if someone tries to compile a model containing such a module, the module can be marked with ``@compile_mode("unsupported")``:

.. jupyter-execute::
    :raises:

    @compile_mode('unsupported')
    class ChildMod(torch.nn.Module):
        pass

    class Supermod(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.child = ChildMod()

    mod = Supermod()
    script(mod)