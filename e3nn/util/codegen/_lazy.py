from typing import List, Optional
import inspect
import textwrap

from e3nn.util import prod

# Indent for code (4 spaces)
_INDENT = "    "


class LazyCodeGenerator:
    def __init__(self,):
        self.blocks = []
        self._reset_state()

    def _reset_state(self):
        # some state needs to be reset each time we generate
        self.indent_level = 0

    def indent(self):
        def f(lazy_codegen):
            lazy_codegen.indent_level += 1
        self(f)

    def dedent(self):
        def f(lazy_codegen):
            lazy_codegen.indent_level -= 1
            assert self.indent_level >= 0
        self(f)

    def __call__(self, b):
        self.blocks.append(b)

    def generate(self):
        self._reset_state()
        # the final lines
        processed_lines = []
        for b in self.blocks:
            if callable(b):
                sig = inspect.signature(b)
                b_kwargs = {
                    'lazy_codegen': self,
                }
                b_kwargs = {k: v for k, v in b_kwargs.items() if k in sig.parameters}
                b = b(**b_kwargs)
            if b is None:
                continue
            # Indent to curent indent
            b = textwrap.indent(b, _INDENT*self.indent_level)
            processed_lines.append(b)
        out = "\n".join(processed_lines)

        return out

    def einsum(self, *args, **kwargs):
        """Generate an einsum."""
        self(LazyEinsum(*args, **kwargs))

    def scalar_multiply(self, x: str, mul: float, out_var: str):
        if isinstance(self.blocks[-1], LazyEinsum):
            lazyein: LazyEinsum = self.blocks[-1]
            if lazyein.out_var == x:
                # We're multiplying the output of an einsum that just happened, so we can incorporate into that einsum:
                lazyein.add_multiplicative_const(mul)
                self(f"{out_var} = {x}")
                return
        self(f"{out_var} = {x}.mul({mul})")

    def script_decorator(self):
        """Insert an ``@torch.jit.script`` decorator."""
        self("@torch.jit.script")


class LazyEinsum:
    einstr: str
    operands: List[str]
    out_var: str
    mul_consts: Optional[List[float]]
    div_consts: Optional[List[float]]
    optimize_einsums: bool

    def __init__(
        self,
        einstr,
        *args,
        out_var="_ein_out",
        mul_consts=None,
        div_consts=None,
    ):
        self.einstr = einstr
        self.operands = args
        self.out_var = out_var
        if isinstance(mul_consts, list):
            self.mul_consts = mul_consts
        elif mul_consts is None:
            self.mul_consts = None
        else:
            self.mul_consts = [mul_consts]
        if isinstance(div_consts, list):
            self.div_consts = div_consts
        elif div_consts is None:
            self.div_consts = None
        else:
            self.div_consts = [div_consts]

    def _get_multiplicitive_const(self):
        # They're already lists
        mul_const = prod(self.mul_consts) if self.mul_consts is not None else None
        div_const = prod(self.div_consts) if self.div_consts is not None else None

        # If we have both multiplicitive and divisor, incorporate
        if mul_const is not None and div_const is not None:
            mul_const = mul_const / div_const
            div_const = None
        elif div_const is not None:
            # If we have only divisor, still take reciprocal and make it a multiplier
            mul_const = 1. / div_const
            div_const = None
        # Be sure that it got incorporated into the multiplicitive constant
        assert div_const is None
        return mul_const

    def add_multiplicative_const(self, mul: float):
        if self.mul_consts is None:
            self.mul_consts = [mul]
        else:
            self.mul_consts.append(mul)

    def __call__(self):
        out = f"{self.out_var} = torch.einsum('{self.einstr}', {', '.join(self.operands)})"
        mul_const = self._get_multiplicitive_const()
        if mul_const is not None:
            out += f".mul({mul_const})"
        return out
