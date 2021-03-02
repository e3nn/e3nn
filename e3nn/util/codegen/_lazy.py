import inspect
import textwrap
import contextvars
import contextlib

from e3nn.util import prod
from ._einsum import opt_einsum_code


# dict mapping id(lazy_codegen_obj) to dict of ids -> info
_ProfileData = contextvars.ContextVar("LazyCodeProfileData")


@contextlib.contextmanager
def profile():
    token = _ProfileData.set({})
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

    def indent(self):
        def f(self):
            self.indent_level += 1
        self(f)

    def dedent(self):
        def f(self):
            self.indent_level -= 1
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
            # initialize the profile data for this object
            processed_lines.append(f"_ProfileData.get()[{id(self)}] = dict()")
        for b in self.blocks:
            if callable(b):
                sig = inspect.signature(b)
                b_kwargs = {
                    'self': self,
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
        return out

    def einsum(self, einstr, *args, out_var="_ein_out", mul_const=None, div_const=None):
        """Generate an einsum."""
        my_einsum_id = self._einsum_id
        self._einsum_id += 1

        # Combine multiple scalar multiples/divisors
        if mul_const is not None and not isinstance(mul_const, float):
            mul_const = prod(mul_const)
        if div_const is not None and not isinstance(div_const, float):
            div_const = prod(mul_const)
        # If we have both multiplicitive and divisor, incorporate
        if mul_const is not None and div_const is not None:
            mul_const = mul_const / div_const
            div_const = None
        elif div_const is not None:
            # If we have only divisor, still take reciprocal and make it a multiplier
            mul_const = 1. / div_const
            div_const = None

        def func_no_opt():
            out = f"{out_var} = torch.einsum('{einstr}', {', '.join(args)})"
            if mul_const is not None:
                out += f".mul({mul_const})"
            if div_const is not None:
                out += f".div({div_const})"
            return out

        if not self.optimize_einsums:
            # Just output a normal einsum
            func = func_no_opt
        else:
            def func(self, profile: bool):
                if profile:
                    # record shapes
                    prof_line = f"_ProfileData.get()[{id(self)}][{my_einsum_id}] = ({', '.join(f'({arg}).shape' for arg in args)})"
                    # normal einsum
                    return prof_line + "\n" + func_no_opt()
                else:
                    # Check if there is profile data to use for optimization
                    profile_dat = _ProfileData.get({})
                    if id(self) in profile_dat and my_einsum_id in profile_dat[id(self)]:
                        arg_shapes = profile_dat[id(self)][my_einsum_id]
                        assert len(arg_shapes) == len(args)
                        # there actually is a profile, use it to optimize
                        return opt_einsum_code(
                            einstr,
                            args,
                            arg_shapes,
                            out_var=out_var
                        )
                    else:
                        # There's no profile data for this einsum, use unoptimized
                        return func_no_opt()

        self(func)

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
