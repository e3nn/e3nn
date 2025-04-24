import torch
import sympy
from sympy.printing.pycode import pycode

from e3nn import o3


def _generate_spherical_harmonics(lmax, device=None) -> None:  # pragma: no cover
    r"""code used to generate the code above

    based on `wigner_3j`
    """
    torch.set_default_dtype(torch.float64)

    def to_frac(x: float):
        from fractions import Fraction

        s = 1 if x >= 0 else -1
        x = x**2
        x = Fraction(x).limit_denominator()
        x = s * sympy.sqrt(x)
        x = sympy.simplify(x)
        return x

    print("sh_0_0 = torch.ones_like(x)")
    print("if lmax == 0:")
    print("    return torch.stack([")
    print("        sh_0_0,")
    print("    ], dim=-1)")
    print()

    x_var, y_var, z_var = sympy.symbols("x y z")
    polynomials = [sympy.sqrt(3) * x_var, sympy.sqrt(3) * y_var, sympy.sqrt(3) * z_var]

    def sub_z1(p, names, polynormz):
        p = p.subs(x_var, 0).subs(y_var, 1).subs(z_var, 0)
        for n, c in zip(names, polynormz):
            p = p.subs(n, c)
        return p

    poly_evalz = [sub_z1(p, [], []) for p in polynomials]

    for l in range(1, lmax + 1):
        sh_variables = sympy.symbols(" ".join(f"sh_{l}_{m}" for m in range(2 * l + 1)))

        for n, p in zip(sh_variables, polynomials):
            print(f"{n} = {pycode(p)}")

        print(f"if lmax == {l}:")
        u = ",\n        ".join(", ".join(f"sh_{j}_{m}" for m in range(2 * j + 1)) for j in range(l + 1))
        print(f"    return torch.stack([\n        {u}\n    ], dim=-1)")
        print()

        if l == lmax:
            break

        polynomials = [
            sum(to_frac(c.item()) * v * sh for cj, v in zip(cij, [x_var, y_var, z_var]) for c, sh in zip(cj, sh_variables))
            for cij in o3.wigner_3j(l + 1, 1, l, device=device)
        ]

        poly_evalz = [sub_z1(p, sh_variables, poly_evalz) for p in polynomials]
        norm = sympy.sqrt(sum(p**2 for p in poly_evalz))
        polynomials = [sympy.sqrt(2 * l + 3) * p / norm for p in polynomials]
        poly_evalz = [sympy.sqrt(2 * l + 3) * p / norm for p in poly_evalz]

        polynomials = [sympy.simplify(p, full=True) for p in polynomials]
