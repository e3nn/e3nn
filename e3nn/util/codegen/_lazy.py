from typing import List, Optional
import inspect
import textwrap
import contextvars
import contextlib
import collections
import logging
logger = logging.getLogger(__name__)

from e3nn.util import prod
from ._einsum import opt_einsum_code


# dict mapping id(lazy_codegen_obj) to dict of ids -> info
_ProfileData = contextvars.ContextVar("LazyCodeProfileData")


@contextlib.contextmanager
def profile():
    # defaultdict so that accesses for previously unset codegen object ids just give an empty dict to start filling
    token = _ProfileData.set(collections.defaultdict(dict))
    try:
        yield
    finally:
        _ProfileData.reset(token)


# Indent for code (4 spaces)
_INDENT = "    "


class LazyCodeGenerator:
    def __init__(
        self,
        optimize_einsums: bool = True
    ):
        #params
        self.optimize_einsums = optimize_einsums
        # state vars
        self.blocks = []
        self._einsum_id = 0
        self._reset_state()

    def _reset_state(self):
        # some state needs to be reset each time we generate
        self.indent_level = 0
        self._ein_naive_cost = 0
        self._ein_opt_cost = 0

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

    def generate(self, profile: bool = False):
        self._reset_state()
        # the final lines
        processed_lines = []
        # If we're profiling, we have to import this module in the code gen
        if profile:
            processed_lines.append("from e3nn.util.codegen._lazy import _ProfileData")
        for b in self.blocks:
            if callable(b):
                sig = inspect.signature(b)
                b_kwargs = {
                    'lazy_codegen': self,
                    'profile': profile
                }
                b_kwargs = {k: v for k, v in b_kwargs.items() if k in sig.parameters}
                b = b(**b_kwargs)
            if b is None:
                continue
            # Indent to curent indent
            b = textwrap.indent(b, _INDENT*self.indent_level)
            processed_lines.append(b)
        out = "\n".join(processed_lines)

        if self._ein_naive_cost > 0:
            logger.debug(
                "einsum optimization: naive cost %.3E; opt cost %.3E (theoretical speedup %.3E)",
                self._ein_naive_cost,
                self._ein_opt_cost,
                self._ein_naive_cost / self._ein_opt_cost
            )
            del self._ein_opt_cost
            del self._ein_naive_cost

        return out

    def einsum(self, *args, **kwargs):
        """Generate an einsum."""
        my_einsum_id = self._einsum_id
        self._einsum_id += 1
        func = LazyEinsum(my_einsum_id, *args, **kwargs)
        self(func)

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
        """Insert an ``@torch.jit.script`` decorator.

        In profiling mode, this does nothing.
        """
        def func(profile: bool):
            if profile:
                return None
            else:
                return "@torch.jit.script"
        self(func)


class LazyEinsum:
    einsum_id: int
    einstr: str
    operands: List[str]
    out_var: str
    mul_consts: Optional[List[float]]
    div_consts: Optional[List[float]]
    optimize_einsums: bool

    def __init__(
        self,
        einsum_id: int,
        einstr,
        *args,
        out_var="_ein_out",
        mul_consts=None,
        div_consts=None,
        optimize_einsums=True
    ):
        self.einsum_id = einsum_id
        self.einstr = einstr
        self.operands = args
        self.out_var = out_var
        self.optimize_einsums = optimize_einsums
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
        mul_const = self.mul_consts
        div_const = self.div_consts
        # Combine multiple scalar multiples/divisors
        if mul_const is not None and not isinstance(mul_const, float):
            mul_const = prod(mul_const)
        if div_const is not None and not isinstance(div_const, float):
            div_const = prod(div_const)
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

    def _ein_no_opt(self):
        out = f"{self.out_var} = torch.einsum('{self.einstr}', {', '.join(self.operands)})"
        mul_const = self._get_multiplicitive_const()
        if mul_const is not None:
            out += f".mul({mul_const})"
        return out

    def add_multiplicative_const(self, mul: float):
        if self.mul_consts is None:
            self.mul_consts = [mul]
        else:
            self.mul_consts.append(mul)

    def __call__(self, lazy_codegen: LazyCodeGenerator, profile: bool):
        if not self.optimize_einsums:
            return self._ein_no_opt()
        codegen_id = id(lazy_codegen)
        if profile:
            # record shapes
            prof_line = f"_ProfileData.get()[{codegen_id}][{self.einsum_id}] = ({', '.join(f'({arg}).shape' for arg in self.operands)})"
            # normal einsum
            return prof_line + "\n" + self._ein_no_opt()
        else:
            # Check if there is profile data to use for optimization
            profile_dat = _ProfileData.get({})
            if codegen_id in profile_dat and self.einsum_id in profile_dat[codegen_id]:
                arg_shapes = profile_dat[codegen_id][self.einsum_id]
                assert len(arg_shapes) == len(self.operands)
                # there actually is a profile, use it to optimize
                mul_const = self._get_multiplicitive_const()
                code, opt_path = opt_einsum_code(
                    self.einstr,
                    self.operands,
                    arg_shapes,
                    out_var=self.out_var,
                    mul_const=mul_const,
                    optimize='optimal'
                )
                lazy_codegen._ein_naive_cost += opt_path.naive_cost
                lazy_codegen._ein_opt_cost += opt_path.opt_cost
                return code
            else:
                # There's no profile data for this einsum, use unoptimized
                return self._ein_no_opt()
