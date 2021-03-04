import textwrap
import torch

from e3nn.util.codegen import LazyCodeGenerator, eval_code


def test_lazy_codegen():
    cg = LazyCodeGenerator()
    vars = []
    def print_vars():
        return f"print({', '.join(vars)})"
    cg("def f():")
    cg.indent()
    cg("x1 = 7")
    vars.append("x1")
    cg("x2 = x1 * 2")
    vars.append("x2")
    cg(print_vars)
    cg.dedent()
    cg("print('hi')")

    out_code = cg.generate()
    true_code = textwrap.dedent("""
    def f():
        x1 = 7
        x2 = x1 * 2
        print(x1, x2)
    print('hi')
    """).strip()
    assert out_code == true_code


def test_einsums():
    einstr = "ij,ij->i"
    mul_const = 4.7
    div_const = 1.1
    cg = LazyCodeGenerator()
    cg("import torch")
    cg("def f(x1, x2):")
    cg.indent()
    cg.einsum(einstr, "x1", "x2", out_var="thing", mul_consts=mul_const, div_consts=div_const)
    cg("return thing")
    cg_func = eval_code(cg.generate()).f

    x1 = torch.randn(3, 4)
    x2 = torch.randn(3, 4)
    assert torch.allclose(
        cg_func(x1, x2), 
        torch.einsum(einstr, x1, x2).mul(mul_const).div(div_const)
    )
