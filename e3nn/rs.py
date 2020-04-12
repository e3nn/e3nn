# pylint: disable=not-callable, no-member, invalid-name, line-too-long, unexpected-keyword-arg, too-many-lines, redefined-builtin
"""
Some functions related to SO3 and his usual representations

Using ZYZ Euler angles parametrisation
"""

from fractions import gcd
from functools import reduce

import torch

from e3nn import o3


def rep(Rs, alpha, beta, gamma, parity=None):
    """
    Representation of O(3). Parity applied (-1)**parity times.
    """
    abc = [alpha, beta, gamma]
    if parity is None:
        return o3.direct_sum(*[o3.irr_repr(l, *abc) for mul, l, _ in simplify(Rs) for _ in range(mul)])
    else:
        assert all(parity != 0 for _, _, parity in simplify(Rs))
        return o3.direct_sum(*[(p ** parity) * o3.irr_repr(l, *abc) for mul, l, p in simplify(Rs) for _ in range(mul)])


def randn(*size, normalization='component'):
    """
    random tensor of representation Rs
    """
    di = 0
    Rs = None
    for di, Rs in enumerate(size):
        if isinstance(Rs, list):
            lsize = size[:di]
            Rs = convention(Rs)
            rsize = size[di + 1:]
            break

    if normalization == 'component':
        return torch.randn(*lsize, dim(Rs), *rsize)
    if normalization == 'norm':
        x = torch.zeros(*lsize, dim(Rs), *rsize)
        start = 0
        for mul, l, _p in Rs:
            r = torch.randn(*lsize, mul, 2 * l + 1, *rsize)
            r.div_(r.norm(2, dim=di + 1, keepdim=True))
            x.narrow(di, start, mul * (2 * l + 1)).copy_(r.view(*lsize, -1, *rsize))
            start += mul * (2 * l + 1)
        return x
    assert False, "normalization needs to be 'norm' or 'component'"


def haslinearpath(Rs_in, l_out, p_out, selection_rule=o3.selection_rule):
    """
    :param Rs_in: list of triplet (multiplicity, representation order, parity)
    :return: if there is a linear operation between them
    """
    for mul_in, l_in, p_in in Rs_in:
        if mul_in == 0:
            continue

        for l in selection_rule(l_in, p_in, l_out, p_out):
            if p_out in (0, p_in * (-1) ** l):
                return True
    return False


def split(Rs, cmul=-1):
    """
    :param Rs: [(mul, 0), (mul, 1), (mul, 2)]
    :return:   mul * [(1, 0), (1, 1), (1, 2)]
    """
    Rs = simplify(Rs)
    muls = {mul for mul, _, _ in Rs}
    if cmul == -1:
        cmul = reduce(gcd, muls)
    assert all(mul % cmul == 0 for mul, _, _ in Rs)

    return cmul * [(mul // cmul, l, p) for mul, l, p in Rs]


def rearrange(Rs_in, Rs_out):
    """
    :return: permutation_matrix

    >>> rearrange([(1, 0), (1, 1)], [(1, 1), (1, 0)])
    tensor([[0., 1., 0., 0.],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.],
            [1., 0., 0., 0.]])

    Example usage:
    permuted_input = einsum('ij,j->i', rearrange(Rs_in, Rs_out), input)
    """
    Rs_in, a = sort(Rs_in)
    Rs_out, b = sort(Rs_out)
    assert simplify(Rs_in) == simplify(Rs_out)
    return b.T @ a


def sort(Rs):
    """
    :return: (Rs_out, permutation_matrix)
    stable sorting of the representation by (l, p)

    >>> sort([(1, 1), (1, 0)])
    ([(1, 0, 0), (1, 1, 0)],
    tensor([[0., 0., 0., 1.],
            [1., 0., 0., 0.],
            [0., 1., 0., 0.],
            [0., 0., 1., 0.]]))

    Example usage:
    sortedRs, permutation_matrix = sort(Rs)
    permuted_input = einsum('ij,j->i', permutation_matrix, input)
    """
    Rs_in = simplify(Rs)
    xs = []

    j = 0  # input offset
    for mul, l, p in Rs_in:
        d = mul * (2 * l + 1)
        xs.append((l, p, mul, j, d))
        j += d

    permutation_matrix = torch.zeros(j, j)

    Rs_out = []
    i = 0  # output offset
    for l, p, mul, j, d in sorted(xs):
        Rs_out.append((mul, l, p))
        permutation_matrix[i:i + d, j:j + d] = torch.eye(d)
        i += d

    return Rs_out, permutation_matrix


def irrep_dim(Rs):
    """
    :param Rs: list of triplet (multiplicity, representation order, [parity])
    :return: number of irreps of the representation without multiplicities
    """
    Rs = convention(Rs)
    return sum(2 * l + 1 for _, l, _ in Rs)


def mul_dim(Rs):
    """
    :param Rs: list of triplet (multiplicity, representation order, [parity])
    :return: number of multiplicities of the representation
    """
    Rs = convention(Rs)
    return sum(mul for mul, _, _ in Rs)


def dim(Rs):
    """
    :param Rs: list of triplet (multiplicity, representation order, [parity])
    :return: dimention of the representation
    """
    Rs = convention(Rs)
    return sum(mul * (2 * l + 1) for mul, l, _ in Rs)


def convention(Rs):
    """
    :param Rs: list of triplet (multiplicity, representation order, [parity])
    :return: conventional version of the same list which always includes parity
    """
    out = []
    for r in Rs:
        if len(r) == 2:
            mul, l = r
            p = 0
        if len(r) == 3:
            mul, l, p = r
            mul = round(mul)
            assert mul >= 0
            if p > 0:
                p = 1
            if p < 0:
                p = -1

        out.append((mul, l, p))
    return out


def simplify(Rs):
    """
    :param Rs: list of triplet (multiplicity, representation order, [parity])
    :return: An equivalent list with parity = {-1, 0, 1} and neighboring orders consolidated into higher multiplicity.

    Note that simplify does not sort the Rs.
    >>> simplify([(1, 1), (1, 1), (1, 0)])
    [(2, 1, 0), (1, 0, 0)]

    Same order Rs which are seperated from each other are not combined
    >>> simplify([(1, 1), (1, 1), (1, 0), (1, 1)])
    [(2, 1, 0), (1, 0, 0), (1, 1, 0)]

    Parity is normalized to {-1, 0, 1}
    >>> simplify([(1, 1, -1), (1, 1, 50), (1, 0, 0)])
    [(1, 1, -1), (1, 1, 1), (1, 0, 0)]
    """
    out = []
    Rs = convention(Rs)
    for mul, l, p in Rs:
        if out and out[-1][1:] == (l, p):
            out[-1] = (out[-1][0] + mul, l, p)
        elif mul > 0:
            out.append((mul, l, p))
    return out


def format_Rs(Rs):
    """
    :param Rs: list of triplet (multiplicity, representation order, [parity])
    :return: simplified version of the same list with the parity
    """
    d = {
        0: "",
        1: "+",
        -1: "-",
    }
    return ",".join("{}{}{}".format("{}x".format(mul) if mul > 1 else "", l, d[p]) for mul, l, p in Rs)


def map_irrep_to_Rs(Rs):
    """
    :param Rs: list of triplet (multiplicity, representation order, [parity])
    :return: mapping matrix from irreps to full representation order (Rs) such
    that einsum('ij,j->i', mapping_matrix, irrep_rep) will have channel
    dimension of dim(Rs).

    examples:
    Rs = [(1, 0), (2, 1), (1, 0)] will return a matrix with 0s and 1s with
    shape [dim(Rs), 1 + 3 + 1] such there is only one 1 in each row, but
    there can be multiple 1s per column.

    Rs = [(2, 0), (2, 1)] will return a matrix with 0s and 1s with
    shape [dim(Rs), 1 + 3] such there is only one 1 in each row (rep), but
    there can be multiple 1s per column (irrep).
    """
    Rs = convention(Rs)
    mapping_matrix = torch.zeros(dim(Rs), irrep_dim(Rs))
    start_irrep = 0
    start_rep = 0
    for mult, L, _ in Rs:
        for _ in range(mult):
            irrep_slice = slice(start_irrep, start_irrep + 2 * L + 1)
            rep_slice = slice(start_rep, start_rep + 2 * L + 1)
            mapping_matrix[rep_slice, irrep_slice] = torch.eye(2 * L + 1)
            start_rep += 2 * L + 1
        start_irrep += 2 * L + 1
    return mapping_matrix  # [dim(Rs), irrep_dim(Rs)]


def map_mul_to_Rs(Rs):
    """
    :param Rs: list of triplet (multiplicity, representation order, [parity])
    :return: mapping matrix for multiplicity and full representation order (Rs)
    such that einsum('ij,j->i', mapping_matrix, mul_rep) will have channel
    dimension of dim(Rs)

    examples:
    Rs = [(1, 0), (2, 1), (1, 0)] will return a matrix with 0s and 1s with
    shape [dim(Rs), 1 + 2 + 1] such there is only one 1 in each row, but
    there can be multiple 1s per column.

    Rs = [(2, 0), (2, 1)] will return a matrix with 0s and 1s with
    shape [dim(Rs), 2 + 2] such there is only one 1 in each row (rep), but
    there can be multiple 1s per column (irrep).
    """
    Rs = convention(Rs)
    mapping_matrix = torch.zeros(dim(Rs), mul_dim(Rs))
    start_mul = 0
    start_rep = 0
    for mult, L, _ in Rs:
        for _ in range(mult):
            rep_slice = slice(start_rep, start_rep + 2 * L + 1)
            mapping_matrix[rep_slice, start_mul] = 1.0
            start_mul += 1
            start_rep += 2 * L + 1
    return mapping_matrix  # [dim(Rs), mul_dim(Rs)]


def tensor_product(input1, input2, output, normalization='component', sorted=False):
    """
    Compute the matrix Q
    from Rs_out to Rs_in1 tensor product with Rs_in2

    For normalization='component',
    The set of "lines" { Q[i] }_i is orthonormal

    :return: Rs_missing, Q

    examples:
    Rs_out, Q = tensor_product_in_in(Rs_in1, Rs_in2, selection_rule)
    # Rs_in1 x Rs_in2 -> Rs_out
    torch.einsum('kij,i,j->k', Q, A, B)

    Rs_in2, Q = tensor_product_in_in(Rs_in1, selection_rule, Rs_out)
    # Rs_in1 x Rs_in2 -> Rs_out
    torch.einsum('kij,i,j->k', Q, A, B)
    """
    if isinstance(input1, list) and isinstance(input2, list):
        return _tensor_product_in_in(input1, input2, output, normalization, sorted)

    if isinstance(input1, list) and isinstance(output, list):
        return _tensor_product_in_out(input1, input2, output, normalization, sorted)

    if isinstance(input2, list) and isinstance(output, list):
        Rs_in1, Q = _tensor_product_in_out(input2, input1, output, normalization, sorted)
        Q = torch.einsum('kij->kji', Q)
        return Rs_in1, Q


def _tensor_product_in_in(Rs_in1, Rs_in2, selection_rule, normalization, sorted):
    """
    Compute the matrix Q
    from Rs_out to Rs_in1 tensor product with Rs_in2
    where Rs_out is a direct sum of irreducible representations

    For normalization='component',
    The set of "lines" { Q[i] }_i is orthonormal

    :return: Rs_out, Q

    example:
    _, Q = tensor_product_in_in(Rs_in1, Rs_in2)
    torch.einsum('kij,i,j->k', Q, A, B)
    """
    assert normalization in ['norm', 'component'], "normalization needs to be 'norm' or 'component'"

    Rs_in1 = simplify(Rs_in1)
    Rs_in2 = simplify(Rs_in2)

    Rs_out = []

    for mul_1, l_1, p_1 in Rs_in1:
        for mul_2, l_2, p_2 in Rs_in2:
            for l_out in selection_rule(l_1, p_1, l_2, p_2):
                Rs_out.append((mul_1 * mul_2, l_out, p_1 * p_2))

    Rs_out = simplify(Rs_out)

    wigner_3j_tensor = torch.zeros(dim(Rs_out), dim(Rs_in1), dim(Rs_in2))

    index_out = 0

    index_1 = 0
    for mul_1, l_1, p_1 in Rs_in1:
        dim_1 = mul_1 * (2 * l_1 + 1)

        index_2 = 0
        for mul_2, l_2, p_2 in Rs_in2:
            dim_2 = mul_2 * (2 * l_2 + 1)
            for l_out in selection_rule(l_1, p_1, l_2, p_2):
                dim_out = mul_1 * mul_2 * (2 * l_out + 1)
                C = o3.wigner_3j(l_out, l_1, l_2, cached=True)
                if normalization == 'component':
                    C *= (2 * l_out + 1) ** 0.5
                if normalization == 'norm':
                    C *= (2 * l_1 + 1) ** 0.5 * (2 * l_2 + 1) ** 0.5
                I = torch.eye(mul_1 * mul_2).view(mul_1 * mul_2, mul_1, mul_2)
                m = torch.einsum("wuv,kij->wkuivj", I, C).view(dim_out, dim_1, dim_2)
                wigner_3j_tensor[index_out:index_out + dim_out, index_1:index_1 + dim_1, index_2:index_2 + dim_2] = m

                index_out += dim_out
            index_2 += dim_2
        index_1 += dim_1

    if sorted:
        Rs_out, perm = sort(Rs_out)
        Rs_out = simplify(Rs_out)
        wigner_3j_tensor = torch.einsum('ij,jkl->ikl', perm, wigner_3j_tensor)

    return Rs_out, wigner_3j_tensor


def _tensor_product_in_out(Rs_in1, selection_rule, Rs_out, normalization, sorted):
    """
    Compute the matrix Q
    from Rs_out to Rs_in1 tensor product with Rs_in2
    where Rs_in2 is a direct sum of irreducible representations

    For normalization='component',
    The set of "lines" { Q[i] }_i is orthonormal

    :return: Rs_in2, Q

    example:
    _, Q = tensor_product_in_out(Rs_in1, Rs_out)
    torch.einsum('kij,i,j->k', Q, A, B)
    """
    assert normalization in ['norm', 'component'], "normalization needs to be 'norm' or 'component'"

    Rs_in1 = simplify(Rs_in1)
    Rs_out = simplify(Rs_out)

    Rs_in2 = []

    for mul_out, l_out, p_out in Rs_out:
        for mul_1, l_1, p_1 in Rs_in1:
            for l_2 in selection_rule(l_1, p_1, l_out, p_out):
                Rs_in2.append((mul_1 * mul_out, l_2, p_1 * p_out))

    Rs_in2 = simplify(Rs_in2)

    wigner_3j_tensor = torch.zeros(dim(Rs_out), dim(Rs_in1), dim(Rs_in2))

    index_2 = 0

    index_out = 0
    for mul_out, l_out, p_out in Rs_out:
        dim_out = mul_out * (2 * l_out + 1)

        n_path = 0
        for mul_1, l_1, p_1 in Rs_in1:
            for l_2 in selection_rule(l_1, p_1, l_out, p_out):
                n_path += mul_1

        index_1 = 0
        for mul_1, l_1, p_1 in Rs_in1:
            dim_1 = mul_1 * (2 * l_1 + 1)
            for l_2 in selection_rule(l_1, p_1, l_out, p_out):
                dim_2 = mul_1 * mul_out * (2 * l_2 + 1)
                C = o3.wigner_3j(l_out, l_1, l_2, cached=True)
                if normalization == 'component':
                    C *= (2 * l_out + 1) ** 0.5
                if normalization == 'norm':
                    C *= (2 * l_1 + 1) ** 0.5 * (2 * l_2 + 1) ** 0.5
                I = torch.eye(mul_out * mul_1).view(mul_out, mul_1, mul_out * mul_1) / n_path ** 0.5
                m = torch.einsum("wuv,kij->wkuivj", I, C).reshape(dim_out, dim_1, dim_2)
                wigner_3j_tensor[index_out:index_out + dim_out, index_1:index_1 + dim_1, index_2:index_2 + dim_2] = m

                index_2 += dim_2
            index_1 += dim_1
        index_out += dim_out

    if sorted:
        Rs_in2, perm = sort(Rs_in2)
        Rs_in2 = simplify(Rs_in2)
        wigner_3j_tensor = torch.einsum('jl,kil->kij', perm, wigner_3j_tensor)

    return Rs_in2, wigner_3j_tensor


def tensor_square(Rs_in, selection_rule=o3.selection_rule, normalization='component', sorted=False):
    """
    Compute the matrix Q
    from Rs_out to Rs_in tensor product with Rs_in
    where Rs_out is a direct sum of irreducible representations

    For normalization='component',
    The set of "lines" { Q[i] }_i is orthonormal

    :return: Rs_out, Q

    example:
    _, Q = tensor_square(Rs_in)
    torch.einsum('kij,i,j->k', Q, A, A)
    """
    assert normalization in ['norm', 'component'], "normalization needs to be 'norm' or 'component'"

    Rs_in = simplify(Rs_in)

    Rs_out = []

    for i, (mul_1, l_1, p_1) in enumerate(Rs_in):
        for l_out in selection_rule(l_1, p_1, l_1, p_1):
            if l_out % 2 == 0:
                Rs_out.append((mul_1 * (mul_1 + 1) // 2, l_out, p_1**2))
            else:
                Rs_out.append((mul_1 * (mul_1 - 1) // 2, l_out, p_1**2))

        for mul_2, l_2, p_2 in Rs_in[i + 1:]:
            for l_out in selection_rule(l_1, p_1, l_2, p_2):
                Rs_out.append((mul_1 * mul_2, l_out, p_1 * p_2))

    Rs_out = simplify(Rs_out)

    wigner_3j_tensor = torch.zeros(dim(Rs_out), dim(Rs_in), dim(Rs_in))

    index_out = 0

    index_1 = 0
    for i, (mul_1, l_1, p_1) in enumerate(Rs_in):
        dim_1 = mul_1 * (2 * l_1 + 1)

        for l_out in selection_rule(l_1, p_1, l_1, p_1):
            I = torch.eye(mul_1**2).view(mul_1**2, mul_1, mul_1)
            uv = I.nonzero()[:, 1:]
            if l_out % 2 == 0:
                I = I[uv[:, 0] <= uv[:, 1]]
            else:
                I = I[uv[:, 0] < uv[:, 1]]

            if I.shape[0] == 0:
                continue

            C = o3.wigner_3j(l_out, l_1, l_1)
            if normalization == 'component':
                C *= (2 * l_out + 1) ** 0.5
            if normalization == 'norm':
                C *= (2 * l_1 + 1) ** 0.5 * (2 * l_1 + 1) ** 0.5
            dim_out = I.shape[0] * (2 * l_out + 1)
            m = torch.einsum("wuv,kij->wkuivj", I, C).reshape(dim_out, dim_1, dim_1)
            wigner_3j_tensor[index_out:index_out + dim_out, index_1:index_1 + dim_1, index_1:index_1 + dim_1] = m

            index_out += dim_out

        index_2 = index_1 + dim_1
        for mul_2, l_2, p_2 in Rs_in[i + 1:]:
            dim_2 = mul_2 * (2 * l_2 + 1)
            for l_out in selection_rule(l_1, p_1, l_2, p_2):
                I = torch.eye(mul_1 * mul_2).view(mul_1 * mul_2, mul_1, mul_2)

                C = o3.wigner_3j(l_out, l_1, l_2)
                if normalization == 'component':
                    C *= (2 * l_out + 1) ** 0.5
                if normalization == 'norm':
                    C *= (2 * l_1 + 1) ** 0.5 * (2 * l_2 + 1) ** 0.5
                dim_out = I.shape[0] * (2 * l_out + 1)
                m = torch.einsum("wuv,kij->wkuivj", I, C).reshape(dim_out, dim_1, dim_2)
                wigner_3j_tensor[index_out:index_out + dim_out, index_1:index_1 + dim_1, index_2:index_2 + dim_2] = m

                index_out += dim_out
            index_2 += dim_2
        index_1 += dim_1

    if sorted:
        Rs_out, perm = sort(Rs_out)
        Rs_out = simplify(Rs_out)
        wigner_3j_tensor = torch.einsum('ij,jkl->ikl', perm, wigner_3j_tensor)
    return Rs_out, wigner_3j_tensor


def elementwise_tensor_product(Rs_1, Rs_2, selection_rule=o3.selection_rule, normalization='component'):
    """
    :return: Rs_out, matrix

    m_kij A_i B_j
    """
    assert normalization in ['norm', 'component'], "normalization needs to be 'norm' or 'component'"

    Rs_1 = simplify(Rs_1)
    Rs_2 = simplify(Rs_2)

    assert sum(mul for mul, _, _ in Rs_1) == sum(mul for mul, _, _ in Rs_2)

    i = 0
    while i < len(Rs_1):
        mul_1, l_1, p_1 = Rs_1[i]
        mul_2, l_2, p_2 = Rs_2[i]

        if mul_1 < mul_2:
            Rs_2[i] = (mul_1, l_2, p_2)
            Rs_2.insert(i + 1, (mul_2 - mul_1, l_2, p_2))

        if mul_2 < mul_1:
            Rs_1[i] = (mul_2, l_1, p_1)
            Rs_1.insert(i + 1, (mul_1 - mul_2, l_1, p_1))
        i += 1

    Rs_out = []
    for (mul, l_1, p_1), (mul_2, l_2, p_2) in zip(Rs_1, Rs_2):
        assert mul == mul_2
        for l in selection_rule(l_1, p_1, l_2, p_2):
            Rs_out.append((mul, l, p_1 * p_2))

    Rs_out = simplify(Rs_out)

    wigner_3j_tensor = torch.zeros(dim(Rs_out), dim(Rs_1), dim(Rs_2))

    index_out = 0
    index_1 = 0
    index_2 = 0
    for (mul, l_1, p_1), (mul_2, l_2, p_2) in zip(Rs_1, Rs_2):
        assert mul == mul_2
        dim_1 = mul * (2 * l_1 + 1)
        dim_2 = mul * (2 * l_2 + 1)

        for l_out in selection_rule(l_1, p_1, l_2, p_2):
            dim_out = mul * (2 * l_out + 1)
            C = o3.wigner_3j(l_out, l_1, l_2, cached=True)
            if normalization == 'component':
                C *= (2 * l_out + 1) ** 0.5
            if normalization == 'norm':
                C *= (2 * l_1 + 1) ** 0.5 * (2 * l_2 + 1) ** 0.5
            I = torch.einsum("uv,wu->wuv", torch.eye(mul), torch.eye(mul))
            m = torch.einsum("wuv,kij->wkuivj", I, C).view(dim_out, dim_1, dim_2)
            wigner_3j_tensor[index_out:index_out + dim_out, index_1:index_1 + dim_1, index_2:index_2 + dim_2] = m
            index_out += dim_out

        index_1 += dim_1
        index_2 += dim_2

    return Rs_out, wigner_3j_tensor
