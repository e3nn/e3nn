import threading
import inspect
import textwrap

from e3nn.util import prod


INDENT = "    "


class LazyCodeGenerator:
    def __init__(
        self,
        optimize_einsums: bool = False
    ):
        #params
        self.optimize_einsums = optimize_einsums
        # state vars
        self.indent_level = 0
        self.blocks = []
        self._thread_local = threading.local()
        self._thread_local.profile = {}

    def indent(self):
        def f():
            self.indent_level += 1
        self(f)

    def dedent(self):
        def f():
            self.indent_level -= 1
            assert self.indent_level >= 0
        self(f)

    def __call__(self, b):
        self.blocks.append(b)

    def einsum(self, einstr, *args, out_var="_ein_out", mul_const=None, div_const=None):
        if mul_const is not None and not isinstance(mul_const, float):
            mul_const = prod(mul_const)
        if div_const is not None and not isinstance(div_const, float):
            div_const = prod(mul_const)
        if mul_const is not None and div_const is not None:
            mul_const = mul_const / div_const
            div_const = None

        if not self.optimize_einsums:
            def func(profile: bool):
                out = f"{out_var} = torch.einsum('{einstr}', {', '.join(args)})"
                if mul_const is not None:
                    out += f".mul({mul_const})"
                if div_const is not None:
                    out += f".div({div_const})"
                return out
        else:
            def func(profile: bool):
                pass
        self(func)

    def generate(self, profile: bool = False):
        processed_lines = []
        for b in self.blocks:
            if callable(b):
                sig = inspect.signature(b)
                if 'profile' in sig.parameters:
                    b = b(profile=profile)
                else:
                    b = b()
            if b is None:
                continue
            # Indent to curent indent
            b = textwrap.indent(b, INDENT*self.indent_level)
            processed_lines.append(b)
        out = "\n".join(processed_lines)
        return out
