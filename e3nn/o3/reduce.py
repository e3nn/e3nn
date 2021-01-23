r"""Function to decompose a multi-index tensor
"""
from e3nn import o3
from e3nn.math import group


def reduce_tensor(formula, eps=1e-9, has_parity=True, **kw_irreps):
    r"""reduce a tensor with symmetries into irreducible representations

    Examples
    --------

    >>> irreps, Q = reduce_tensor('ijkl=jikl=ikjl=ijlk', i="1e")
    >>> irreps
    1x0e+1x2e+1x4e
    """
    gr = group.O3() if has_parity else group.SO3()

    kw_representations = {}

    for i in kw_irreps:
        if callable(kw_irreps[i]):
            kw_representations[i] = kw_irreps[i]
        else:
            if has_parity:
                kw_representations[i] = lambda g: o3.Irreps(kw_irreps[i]).D_from_quaternion(*g)
            else:
                kw_representations[i] = o3.Irreps(kw_irreps[i]).D_from_quaternion

    irreps, Q = group.reduce_tensor(gr, formula, eps, **kw_representations)

    if has_parity:
        irreps = o3.Irreps(irreps)
    else:
        irreps = o3.Irreps([(mul, l, 1) for mul, l in irreps])

    return irreps, Q


def reduce_irreps_tensor(formula, **irreps):
    f0, formulas = group.germinate_formulas(formula)

    irreps = {i: o3.Irreps(irs) for i, irs in irreps.items()}

    for _s, p in formulas:
        f = "".join(f0[i] for i in p)
        for i, j in zip(f0, f):
            if i in irreps and j in irreps and irreps[i] != irreps[j]:
                raise RuntimeError(f'irreps of {i} and {j} should be the same')
            if i in irreps:
                irreps[j] = irreps[i]
            if j in irreps:
                irreps[i] = irreps[j]

    for i in f0:
        if i not in irreps:
            raise RuntimeError(f'index {i} has no irreps associated to it')

    Q, base = group.reduce_permutation(f0, formulas, **{i: irs.dim for i, irs in irreps.items()})
