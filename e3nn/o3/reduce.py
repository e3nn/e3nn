r"""Function to decompose a multi-index tensor
"""
from e3nn import group, o3


def reduce_tensor(formula, eps=1e-9, has_parity=True, **kw_irreps):
    r"""reduce a tensor with symmetries into irreducible representations

    Examples
    --------

    >>> irreps, Q = reduce_tensor('ijkl=jikl=ikjl=ijlk', i="1e")
    >>> irreps
    0e+2e+4e
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
