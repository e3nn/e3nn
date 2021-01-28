import itertools

import torch
from e3nn.math import direct_sum, kron, perm
from e3nn.math.group import Group, has_rep_in_rep, is_representation
from e3nn.util import torch_default_dtype


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


def reduce_permutation(f0, formulas, **dims):

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
    Q = torch.zeros(d_sym, len(full_base))
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


def reduce_tensor(group: Group, formula, eps=1e-9, **kw_representations):
    r"""reduce a tensor with symmetries into irreducible representations
    Usage
    irreps, Q = rs.reduce_tensor('ijkl=jikl=ikjl=ijlk', i=rep)
    irreps = [(mul1, r1), (mul2, r2), ...]
    Q = tensor of shape [15, 3, 3, 3, 3]
    """
    dtype = torch.get_default_dtype()
    with torch_default_dtype(torch.float64):
        f0, formulas = germinate_formulas(formula)

        # here we check that each index has one and only one representation
        for _s, p in formulas:
            f = "".join(f0[i] for i in p)
            for i, j in zip(f0, f):
                if i in kw_representations and j in kw_representations and kw_representations[i] != kw_representations[j]:
                    raise RuntimeError(f'rep of {i} and {j} should be the same')
                if i in kw_representations:
                    kw_representations[j] = kw_representations[i]
                if j in kw_representations:
                    kw_representations[i] = kw_representations[j]

        for i in f0:
            if i not in kw_representations:
                raise RuntimeError(f'index {i} has no representations associated to it')

        ide = group.identity()
        dims = {i: len(kw_representations[i](ide)) for i in f0}  # dimension of each index

        Q, _ = reduce_permutation(f0, formulas, **dims)
        d_sym = len(Q)

        if d_sym == 0:
            return [], torch.zeros(d_sym, *[dims[i] for i in f0]).to(dtype=dtype)

        Q = Q.reshape(len(Q), -1)

        # up to now, the only used information about the representations was their dimensions

        # We project the representation on the basis `base`
        def representation(g):
            m = kron(*(kw_representations[i](g) for i in f0))
            return Q @ m @ Q.T

        # And check that after this projection it is still a representation
        assert is_representation(group, representation, eps)

        # The rest of the code simply extract the irreps present in this representation
        irreps = []
        A = Q.clone()
        for ir in group.irrep_indices():
            if group.rep(ir)(ide).shape[0] > d_sym - sum(mul * group.rep_dim(ir) for mul, ir in irreps):
                break

            mul, B, representation = has_rep_in_rep(group, representation, group.rep(ir), eps)
            A = direct_sum(torch.eye(d_sym - B.shape[0]), B) @ A
            A = _round_sqrt(A, eps)

            if mul > 0:
                irreps += [(mul, ir)]

            if sum(mul * group.rep_dim(ir) for mul, ir in irreps) == d_sym:
                break

        if sum(mul * group.rep_dim(ir) for mul, ir in irreps) != d_sym:
            raise RuntimeError('unable to decompose into irreducible representations')

        A = A.reshape(len(A), *[dims[i] for i in f0])
        return irreps, A.to(dtype=dtype)


def _round_sqrt(x, eps):
    # round off x assuming it contains terms of the form +-1/sqrt(N)
    x[x.abs() < eps] = 0
    x = x.sign() / x.pow(2)
    x = x.div(eps).round().mul(eps)
    x = x.sign() / x.abs().sqrt()
    x[torch.isnan(x)] = 0
    x[torch.isinf(x)] = 0
    return x
