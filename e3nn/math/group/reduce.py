import itertools

import torch
from e3nn.math import perm


def germinate_formulas(formula):
    formulas = [
        (-1 if f.startswith('-') else 1, f.replace('-', ''))
        for f in formula.split('=')
    ]
    s0, f0 = formulas[0]
    assert s0 == 1

    for _s, f in formulas:
        if len(set(f)) != len(f) or set(f) != set(f0):
            raise RuntimeError(f'{f} is not a permutation of {f0}')
        if len(f0) != len(f):
            raise RuntimeError(f'{f0} and {f} don\'t have the same number of indices')

    # `formulas` is a list of (sign, permutation of indices)
    # each formula can be viewed as a permutation of the original formula
    formulas = {(s, tuple(f.index(i) for i in f0)) for s, f in formulas}  # set of generators (permutations)

    # they can be composed, for instance if you have ijk=jik=ikj
    # you also have ijk=jki
    # applying all possible compositions creates an entire group
    while True:
        n = len(formulas)
        formulas = formulas.union([(s, perm.inverse(p)) for s, p in formulas])
        formulas = formulas.union([
            (s1 * s2, perm.compose(p1, p2))
            for s1, p1 in formulas
            for s2, p2 in formulas
        ])
        if len(formulas) == n:
            break  # we break when the set is stable => it is now a group \o/

    return f0, formulas


def reduce_permutation(f0, formulas, dtype=None, device=None, **dims):

    # here we check that each index has one and only one dimension
    for _s, p in formulas:
        f = "".join(f0[i] for i in p)
        for i, j in zip(f0, f):
            if i in dims and j in dims and dims[i] != dims[j]:
                raise RuntimeError(f'dimension of {i} and {j} should be the same')
            if i in dims:
                dims[j] = dims[i]
            if j in dims:
                dims[i] = dims[j]

    for i in f0:
        if i not in dims:
            raise RuntimeError(f'index {i} has no dimension associated to it')

    dims = [dims[i] for i in f0]

    full_base = list(itertools.product(*(range(d) for d in dims)))  # (0, 0, 0), (0, 0, 1), (0, 0, 2), ... (3, 3, 3)
    # len(full_base) degrees of freedom in an unconstrained tensor

    # but there is constraints given by the group `formulas`
    # For instance if `ij=-ji`, then 00=-00, 01=-01 and so on
    base = set()
    for x in full_base:
        # T[x] is a coefficient of the tensor T and is related to other coefficient T[y]
        # if x and y are related by a formula
        xs = {(s, tuple(x[i] for i in p)) for s, p in formulas}
        # s * T[x] are all equal for all (s, x) in xs
        # if T[x] = -T[x] it is then equal to 0 and we lose this degree of freedom
        if not (-1, x) in xs:
            # the sign is arbitrary, put both possibilities
            base.add(frozenset({
                frozenset(xs),
                frozenset({(-s, x) for s, x in xs})
            }))

    # len(base) is the number of degrees of freedom in the tensor.
    # Now we want to decompose these degrees of freedom into irreps

    base = sorted([sorted([sorted(xs) for xs in x]) for x in base])  # requested for python 3.7 but not for 3.8 (probably a bug in 3.7)

    # First we compute the change of basis (projection) between full_base and base
    d_sym = len(base)
    Q = torch.zeros(d_sym, len(full_base), dtype=dtype, device=device)
    ret = []

    for i, x in enumerate(base):
        x = max(x, key=lambda xs: sum(s for s, x in xs))
        ret.append(x)
        for s, e in x:
            # j = full_base.index(e)
            j = 0
            for k, d in zip(e, dims):
                j *= d
                j += k
            Q[i, j] = s / len(x)**0.5

    # assert torch.allclose(Q @ Q.T, torch.eye(d_sym))

    Q = Q.reshape(d_sym, *dims)
    return Q, ret
