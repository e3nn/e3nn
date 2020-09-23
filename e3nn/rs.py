# pylint: disable=not-callable, no-member, invalid-name, line-too-long, unexpected-keyword-arg, too-many-lines, redefined-builtin, arguments-differ, abstract-method
"""
Some functions related to SO3 and his usual representations

Using ZYZ Euler angles parametrisation
"""

import itertools
from functools import partial, reduce
from math import gcd
from typing import List, Tuple, Union

import torch
from torch_sparse import SparseTensor

from e3nn import o3, perm
from e3nn.util.default_dtype import torch_default_dtype
from e3nn.util.sparse import get_sparse_buffer, register_sparse_buffer

TY_RS_LOOSE = Union[List[Union[int, Tuple[int, int], Tuple[int, int, int]]], int]
TY_RS_STRICT = List[Tuple[int, int, int]]


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


def randn(*size, normalization='component', dtype=None, device=None, requires_grad=False):
    """
    random tensor of representation Rs
    """
    di = 0
    Rs = None
    for di, n in enumerate(size):
        if isinstance(n, list):
            if Rs is not None:
                raise RuntimeError('Only one Rs is allowed')
            lsize = size[:di]
            Rs = convention(n)
            rsize = size[di + 1:]
    if Rs is None:
        raise RuntimeError('Rs missing')

    if normalization == 'component':
        return torch.randn(*lsize, dim(Rs), *rsize, dtype=dtype, device=device, requires_grad=requires_grad)
    if normalization == 'norm':
        x = torch.zeros(*lsize, dim(Rs), *rsize, dtype=dtype, device=device, requires_grad=requires_grad)
        with torch.no_grad():
            start = 0
            for mul, l, _p in Rs:
                r = torch.randn(*lsize, mul, 2 * l + 1, *rsize)
                r.div_(r.norm(2, dim=di + 1, keepdim=True))
                x.narrow(di, start, mul * (2 * l + 1)).copy_(r.reshape(*lsize, -1, *rsize))
                start += mul * (2 * l + 1)
        return x
    assert False, "normalization needs to be 'norm' or 'component'"


def haslinearpath(Rs_in: TY_RS_STRICT, l_out: int, p_out: int, selection_rule: o3.TY_SELECTION_RULE = o3.selection_rule):
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


def transpose_mul(Rs, cmul=-1):
    """
    :param Rs: [(mul, 0), (mul, 1), (mul, 2)]
    :return:   mul * [(1, 0), (1, 1), (1, 2)]
    """
    Rs = simplify(Rs)
    muls = {mul for mul, _, _ in Rs}
    if cmul == -1:
        cmul = reduce(gcd, muls)
    assert all(mul % cmul == 0 for mul, _, _ in Rs)

    return cmul, [(mul // cmul, l, p) for mul, l, p in Rs]


def cut(features, *Rss, dim_=-1):
    """
    Cut `feaures` according to the list of Rs
    """
    index = 0
    outputs = []
    for Rs in Rss:
        n = dim(Rs)
        outputs.append(features.narrow(dim_, index, n))
        index += n
    assert index == features.shape[dim_]
    return outputs


class TransposeToMulL(torch.nn.Module):
    """
    [(mul, 1), (mul, 2)]  ->  mul * [(1, 1), (1, 2)]
    [batch, l * mul * m]  ->  [batch, mul, l * m]
    """
    def __init__(self, Rs):
        super().__init__()
        self.Rs_in = convention(Rs)
        self.mul, self.Rs_out = transpose_mul(self.Rs_in)
        register_sparse_buffer(self, 'mixing_matrix', rearrange(self.Rs_in, self.mul * self.Rs_out))

    def __repr__(self):
        return "{name} ({Rs_in} -> {mul} x {Rs_out})".format(
            name=self.__class__.__name__,
            Rs_in=format_Rs(self.Rs_in),
            mul=self.mul,
            Rs_out=format_Rs(self.Rs_out),
        )

    def forward(self, features):
        *size, n = features.size()
        features = features.reshape(-1, n)

        mixing_matrix = get_sparse_buffer(self, 'mixing_matrix')
        # features = torch.einsum('ij,zj->zi', self.mixing_matrix, features)
        features = (mixing_matrix @ features.T).T
        return features.reshape(*size, self.mul, -1)


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
    return b.t() @ a


def sort(Rs: TY_RS_LOOSE) -> SparseTensor:
    """
    :return: (Rs_out, permutation_matrix)
    stable sorting of the representation by (l, p)

    sorted = perm @ unsorted

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

    index = []

    Rs_out = []
    i = 0  # output offset
    for l, p, mul, j, d in sorted(xs):
        Rs_out.append((mul, l, p))
        for _ in range(d):
            index.append([i, j])
            i += 1
            j += 1

    index = torch.tensor(index).T
    permutation_matrix = SparseTensor(row=index[0], col=index[1], value=torch.ones(index.shape[1]))

    return Rs_out, permutation_matrix


def irrep_dim(Rs: TY_RS_LOOSE) -> int:
    """
    :param Rs: list of triplet (multiplicity, representation order, [parity])
    :return: number of irreps of the representation without multiplicities
    """
    Rs = convention(Rs)
    return sum(2 * l + 1 for _, l, _ in Rs)


def mul_dim(Rs: TY_RS_LOOSE) -> int:
    """
    :param Rs: list of triplet (multiplicity, representation order, [parity])
    :return: number of multiplicities of the representation
    """
    Rs = convention(Rs)
    return sum(mul for mul, _, _ in Rs)


def dim(Rs: TY_RS_LOOSE) -> int:
    """
    :param Rs: list of triplet (multiplicity, representation order, [parity])
    :return: dimention of the representation
    """
    Rs = convention(Rs)
    return sum(mul * (2 * l + 1) for mul, l, _ in Rs)


def lmax(Rs: TY_RS_LOOSE) -> int:
    """
    :param Rs: list of triplet (multiplicity, representation order, [parity])
    :return: maximum l present in the signal
    """
    return max(l for mul, l, _ in convention(Rs) if mul > 0)


def convention(Rs: TY_RS_LOOSE) -> TY_RS_STRICT:
    """
    :param Rs: list of triplet (multiplicity, representation order, [parity])
    :return: conventional version of the same list which always includes parity
    """
    if isinstance(Rs, int):
        return [(1, Rs, 0)]

    out = []
    for r in Rs:
        if isinstance(r, int):
            mul, l, p = 1, r, 0
        elif len(r) == 2:
            (mul, l), p = r, 0
        elif len(r) == 3:
            mul, l, p = r

        assert isinstance(mul, int) and mul >= 0
        assert isinstance(l, int) and l >= 0
        assert p in [0, 1, -1]

        out.append((mul, l, p))
    return out


def simplify(Rs: TY_RS_LOOSE) -> TY_RS_STRICT:
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


def are_equal(Rs1: TY_RS_LOOSE, Rs2: TY_RS_LOOSE) -> bool:
    """
    :param Rs1: first list of triplet (multiplicity, representation order, [parity])
    :param Rs2: second list of triplet (multiplicity, representation order, [parity])

    examples:
    Rs1 = [(1, 0), (1, 0), (1, 0)]
    Rs2 = [(3, 0)]
    are_equal(Rs1, Rs2)
    >> True

    Rs1 = [(1, 0), (1, 1), (1, 0)]
    Rs2 = [(2, 0), (1, 1)]
    are_equal(Rs1, Rs2)
    >> False
    Irreps are not in the same order
    """
    return simplify(Rs1) == simplify(Rs2)


def format_Rs(Rs: TY_RS_LOOSE) -> str:
    """
    :param Rs: list of triplet (multiplicity, representation order, [parity])
    :return: simplified version of the same list with the parity
    """
    Rs = convention(Rs)
    d = {
        0: "",
        1: "e",
        -1: "o",
    }
    return ",".join("{}{}{}".format("{}x".format(mul) if mul > 1 else "", l, d[p]) for mul, l, p in Rs if mul > 0)


def map_irrep_to_Rs(Rs: TY_RS_LOOSE) -> torch.Tensor:
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


def map_mul_to_Rs(Rs: TY_RS_LOOSE) -> torch.Tensor:
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


def tensor_product(
        input1: Union[TY_RS_LOOSE, o3.TY_SELECTION_RULE],
        input2: Union[TY_RS_LOOSE, o3.TY_SELECTION_RULE],
        output: Union[TY_RS_LOOSE, o3.TY_SELECTION_RULE],
        normalization: str = 'component',
        sorted: bool = False
) -> Tuple[TY_RS_STRICT, SparseTensor]:
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
        # [out, in2 * in1] -> [out, in1 * in2]
        row, col, val = Q.coo()
        n1 = dim(Rs_in1)
        n2 = dim(input2)
        col = n2 * (col % n1) + col // n1
        Q = SparseTensor(row=row,
                         col=col,
                         value=val,
                         sparse_sizes=(dim(output), n1 * n2))
        return Rs_in1, Q


class TensorProduct(torch.nn.Module):
    """
    Module for tensor_product
    """
    def __init__(self, input1, input2, output, normalization='component', sorted=True):
        super().__init__()

        Rs, mat = tensor_product(input1, input2, output, normalization, sorted)

        if not isinstance(input1, list):
            self.Rs_in1 = Rs
            self.Rs_in2 = convention(input2)
            self.Rs_out = convention(output)
            self._complete = 'in1'
        if not isinstance(input2, list):
            self.Rs_in1 = convention(input1)
            self.Rs_in2 = Rs
            self.Rs_out = convention(output)
            self._complete = 'in2'
        if not isinstance(output, list):
            self.Rs_in1 = convention(input1)
            self.Rs_in2 = convention(input2)
            self.Rs_out = Rs
            self._complete = 'out'

        register_sparse_buffer(self, 'mixing_matrix', mat)

    def __repr__(self):
        return "{name} ({Rs_in1} x {Rs_in2} -> {Rs_out})".format(
            name=self.__class__.__name__,
            Rs_in1=format_Rs(self.Rs_in1),
            Rs_in2=format_Rs(self.Rs_in2),
            Rs_out=format_Rs(self.Rs_out),
        )

    def forward(self, features_1, features_2=None):
        '''
        :param features_1: [..., in1] or [..., in1, in2] if features_2 is None
        :param features_2: [..., in2]
        :return: [..., out]
        '''
        d_out = dim(self.Rs_out)
        d_in1 = dim(self.Rs_in1)
        d_in2 = dim(self.Rs_in2)

        if self._complete == 'out' or features_2 is None:
            if features_2 is None:
                features = features_1
            else:
                features = features_1[..., :, None] * features_2[..., None, :]

            size = features.shape[:-2]
            features = features.reshape(-1, d_in1, d_in2)  # [in1, in2, batch]

            mixing_matrix = get_sparse_buffer(self, "mixing_matrix")  # [out, in1 * in2]

            features = torch.einsum('zij->ijz', features)  # [in1, in2, batch]
            features = features.reshape(d_in1 * d_in2, features.shape[2])
            features = mixing_matrix @ features  # [out, batch]
            return features.T.reshape(*size, d_out)
        if self._complete == 'in1':
            k = self.left(features_1)  # [..., out, in2]
            return torch.einsum('...ij,...j->...i', k, features_2)
        if self._complete == 'in2':
            k = self.right(features_2)  # [..., out, in1]
            return torch.einsum('...ij,...j->...i', k, features_1)

    def right(self, features_2):
        '''
        :param features_2: [..., in2]
        :return: [..., out, in1]
        '''
        d_out = dim(self.Rs_out)
        d_in1 = dim(self.Rs_in1)
        d_in2 = dim(self.Rs_in2)
        size_2 = features_2.shape[:-1]
        if d_in2 == 0:
            return features_2.new_zeros(*size_2, d_out, d_in1)

        features_2 = features_2.reshape(-1, d_in2)

        mixing_matrix = get_sparse_buffer(self, "mixing_matrix")  # [out, in1 * in2]
        mixing_matrix = mixing_matrix.sparse_reshape(d_out * d_in1, d_in2)
        output = mixing_matrix @ features_2.T  # [out * in1, batch]
        return output.T.reshape(*size_2, d_out, d_in1)

    def left(self, features_1):
        '''
        :param features_1: [..., in1]
        :return: [..., out, in2]
        '''
        d_out = dim(self.Rs_out)
        d_in1 = dim(self.Rs_in1)
        d_in2 = dim(self.Rs_in2)
        size_1 = features_1.shape[:-1]
        if d_in1 == 0:
            return features_1.new_zeros(*size_1, d_out, d_in2)

        features_1 = features_1.reshape(-1, d_in1)

        mixing_matrix = get_sparse_buffer(self, "mixing_matrix")  # [out, in1 * in2]
        mixing_matrix = mixing_matrix.sparse_reshape(d_out * d_in1, d_in2).t()  # [in2, out * in1]
        mixing_matrix = mixing_matrix.sparse_reshape(d_in2 * d_out, d_in1)  # [in2 * out, in1]
        output = mixing_matrix @ features_1.T  # [in2 * out, batch]
        output = output.reshape(d_in2, d_out, features_1.shape[0])
        output = torch.einsum('jiz->zij', output)
        return output.reshape(*size_1, d_out, d_in2)

    def to_dense(self):
        """
        :return: tensor of shape [dim(Rs_out), dim(Rs_in1), dim(Rs_in2)]
        """
        mixing_matrix = get_sparse_buffer(self, "mixing_matrix")  # [out, in1 * in2]
        mixing_matrix = mixing_matrix.to_dense()
        mixing_matrix = mixing_matrix.reshape(dim(self.Rs_out), dim(self.Rs_in1), dim(self.Rs_in2))
        return mixing_matrix


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

    dim_in2 = dim(Rs_in2)
    row = []
    col = []
    val = []

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
                I = torch.eye(mul_1 * mul_2).reshape(mul_1 * mul_2, mul_1, mul_2)
                m = torch.einsum("wuv,kij->wkuivj", I, C).reshape(dim_out, dim_1, dim_2)
                i_out, i_1, i_2 = m.nonzero(as_tuple=False).T
                i_out += index_out
                i_1 += index_1
                i_2 += index_2
                row.append(i_out)
                col.append(i_1 * dim_in2 + i_2)
                val.append(m[m != 0])

                index_out += dim_out
            index_2 += dim_2
        index_1 += dim_1

    wigner_3j_tensor = SparseTensor(
        row=torch.cat(row) if row else torch.zeros(0, dtype=torch.long),
        col=torch.cat(col) if col else torch.zeros(0, dtype=torch.long),
        value=torch.cat(val) if val else torch.zeros(0),
        sparse_sizes=(dim(Rs_out), dim(Rs_in1) * dim(Rs_in2)))

    if sorted:
        Rs_out, perm_mat = sort(Rs_out)
        Rs_out = simplify(Rs_out)
        wigner_3j_tensor = perm_mat @ wigner_3j_tensor

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

    dim_in2 = dim(Rs_in2)
    row = []
    col = []
    val = []

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
                if l_2 == 0:
                    assert l_out == l_1
                    l = l_1
                    dim_2 = mul_1 * mul_out
                    i_out = []
                    i_1 = []
                    i_2 = []
                    v = 0
                    for w in range(mul_out):
                        for u in range(mul_1):
                            i_out += [(2 * l + 1) * w + m for m in range(2 * l + 1)]
                            i_1 += [(2 * l + 1) * u + m for m in range(2 * l + 1)]
                            i_2 += (2 * l + 1) * [v]
                            v += 1
                    i_out = index_out + torch.tensor(i_out)
                    i_1 = index_1 + torch.tensor(i_1)
                    i_2 = index_2 + torch.tensor(i_2)
                    m = torch.ones((2 * l + 1) * dim_2) / n_path ** 0.5
                else:
                    dim_2 = mul_1 * mul_out * (2 * l_2 + 1)
                    C = o3.wigner_3j(l_out, l_1, l_2, cached=True)
                    if normalization == 'component':
                        C *= (2 * l_out + 1) ** 0.5
                    if normalization == 'norm':
                        C *= (2 * l_1 + 1) ** 0.5 * (2 * l_2 + 1) ** 0.5
                    I = torch.eye(mul_out * mul_1).reshape(mul_out, mul_1, mul_out * mul_1) / n_path ** 0.5
                    m = torch.einsum("wuv,kij->wkuivj", I, C).reshape(dim_out, dim_1, dim_2)
                    i_out, i_1, i_2 = m.nonzero(as_tuple=True)  # slow part
                    m = m[(i_out, i_1, i_2)]
                    i_out += index_out
                    i_1 += index_1
                    i_2 += index_2

                row.append(i_out)
                col.append(i_1 * dim_in2 + i_2)
                val.append(m)

                index_2 += dim_2
            index_1 += dim_1
        index_out += dim_out

    wigner_3j_tensor = SparseTensor(
        row=torch.cat(row) if row else torch.zeros(0, dtype=torch.long),
        col=torch.cat(col) if col else torch.zeros(0, dtype=torch.long),
        value=torch.cat(val) if val else torch.zeros(0),
        sparse_sizes=(dim(Rs_out), dim(Rs_in1) * dim(Rs_in2)))

    if sorted:
        Rs_in2, perm_mat = sort(Rs_in2)
        Rs_in2 = simplify(Rs_in2)
        # sorted = perm_mat @ unsorted
        wigner_3j_tensor = wigner_3j_tensor.sparse_reshape(-1, dim(Rs_in2))
        wigner_3j_tensor = wigner_3j_tensor @ perm_mat.t()  # slow part
        wigner_3j_tensor = wigner_3j_tensor.sparse_reshape(-1, dim(Rs_in1) * dim(Rs_in2))

    return Rs_in2, wigner_3j_tensor


def tensor_square(
        Rs_in: TY_RS_LOOSE,
        selection_rule: o3.TY_SELECTION_RULE = o3.selection_rule,
        normalization: str = 'component',
        sorted: bool = False
) -> Tuple[TY_RS_STRICT, SparseTensor]:
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

    dim_in = dim(Rs_in)
    row = []
    col = []
    val = []

    index_out = 0

    index_1 = 0
    for i, (mul_1, l_1, p_1) in enumerate(Rs_in):
        dim_1 = mul_1 * (2 * l_1 + 1)

        for l_out in selection_rule(l_1, p_1, l_1, p_1):
            I = torch.eye(mul_1**2).reshape(mul_1**2, mul_1, mul_1)
            uv = I.nonzero(as_tuple=False)[:, 1:]
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
            i_out, i_1, i_2 = m.nonzero(as_tuple=False).T
            i_out += index_out
            i_1 += index_1
            i_2 += index_1
            row.append(i_out)
            col.append(i_1 * dim_in + i_2)
            val.append(m[m != 0])

            index_out += dim_out

        index_2 = index_1 + dim_1
        for mul_2, l_2, p_2 in Rs_in[i + 1:]:
            dim_2 = mul_2 * (2 * l_2 + 1)
            for l_out in selection_rule(l_1, p_1, l_2, p_2):
                I = torch.eye(mul_1 * mul_2).reshape(mul_1 * mul_2, mul_1, mul_2)

                C = o3.wigner_3j(l_out, l_1, l_2)
                if normalization == 'component':
                    C *= (2 * l_out + 1) ** 0.5
                if normalization == 'norm':
                    C *= (2 * l_1 + 1) ** 0.5 * (2 * l_2 + 1) ** 0.5
                dim_out = I.shape[0] * (2 * l_out + 1)
                m = torch.einsum("wuv,kij->wkuivj", I, C).reshape(dim_out, dim_1, dim_2)
                i_out, i_1, i_2 = m.nonzero(as_tuple=False).T
                i_out += index_out
                i_1 += index_1
                i_2 += index_2
                row.append(i_out)
                col.append(i_1 * dim_in + i_2)
                val.append(m[m != 0])

                index_out += dim_out
            index_2 += dim_2
        index_1 += dim_1

    wigner_3j_tensor = SparseTensor(
        row=torch.cat(row) if row else torch.zeros(0, dtype=torch.long),
        col=torch.cat(col) if col else torch.zeros(0, dtype=torch.long),
        value=torch.cat(val) if val else torch.zeros(0),
        sparse_sizes=(dim(Rs_out), dim(Rs_in) * dim(Rs_in)))

    if sorted:
        Rs_out, perm_mat = sort(Rs_out)
        Rs_out = simplify(Rs_out)
        # sorted = perm_mat @ unsorted
        wigner_3j_tensor = perm_mat @ wigner_3j_tensor

    return Rs_out, wigner_3j_tensor


class TensorSquare(torch.nn.Module):
    """
    Module for tensor_square
    """
    def __init__(self, Rs_in, selection_rule=o3.selection_rule):
        super().__init__()

        self.Rs_in = simplify(Rs_in)

        self.Rs_out, mixing_matrix = tensor_square(self.Rs_in, selection_rule, sorted=True)
        register_sparse_buffer(self, 'mixing_matrix', mixing_matrix)

    def __repr__(self):
        return "{name} ({Rs_in} ^ 2 -> {Rs_out})".format(
            name=self.__class__.__name__,
            Rs_in=format_Rs(self.Rs_in),
            Rs_out=format_Rs(self.Rs_out),
        )

    def forward(self, features):
        '''
        :param features: [..., channels]
        '''
        *size, n = features.size()
        features = features.reshape(-1, n)

        mixing_matrix = get_sparse_buffer(self, "mixing_matrix")

        features = torch.einsum('zi,zj->ijz', features, features)
        features = mixing_matrix @ features.reshape(-1, features.shape[2])
        return features.T.reshape(*size, -1)


def elementwise_tensor_product(
        Rs_in1: TY_RS_LOOSE,
        Rs_in2: TY_RS_LOOSE,
        selection_rule: o3.TY_SELECTION_RULE = o3.selection_rule,
        normalization: str = 'component'
) -> Tuple[TY_RS_STRICT, SparseTensor]:
    """
    :return: Rs_out, matrix

    m_kij A_i B_j
    """
    assert normalization in ['norm', 'component'], "normalization needs to be 'norm' or 'component'"

    Rs_in1 = simplify(Rs_in1)
    Rs_in2 = simplify(Rs_in2)

    assert sum(mul for mul, _, _ in Rs_in1) == sum(mul for mul, _, _ in Rs_in2)

    i = 0
    while i < len(Rs_in1):
        mul_1, l_1, p_1 = Rs_in1[i]
        mul_2, l_2, p_2 = Rs_in2[i]

        if mul_1 < mul_2:
            Rs_in2[i] = (mul_1, l_2, p_2)
            Rs_in2.insert(i + 1, (mul_2 - mul_1, l_2, p_2))

        if mul_2 < mul_1:
            Rs_in1[i] = (mul_2, l_1, p_1)
            Rs_in1.insert(i + 1, (mul_1 - mul_2, l_1, p_1))
        i += 1

    Rs_out = []
    for (mul, l_1, p_1), (mul_2, l_2, p_2) in zip(Rs_in1, Rs_in2):
        assert mul == mul_2
        for l in selection_rule(l_1, p_1, l_2, p_2):
            Rs_out.append((mul, l, p_1 * p_2))

    Rs_out = simplify(Rs_out)

    dim_in2 = dim(Rs_in2)
    row = []
    col = []
    val = []

    index_out = 0
    index_1 = 0
    index_2 = 0
    for (mul, l_1, p_1), (mul_2, l_2, p_2) in zip(Rs_in1, Rs_in2):
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
            m = torch.einsum("wuv,kij->wkuivj", I, C).reshape(dim_out, dim_1, dim_2)
            i_out, i_1, i_2 = m.nonzero(as_tuple=False).T
            i_out += index_out
            i_1 += index_1
            i_2 += index_2
            row.append(i_out)
            col.append(i_1 * dim_in2 + i_2)
            val.append(m[m != 0])

            index_out += dim_out

        index_1 += dim_1
        index_2 += dim_2

    wigner_3j_tensor = SparseTensor(
        row=torch.cat(row) if row else torch.zeros(0, dtype=torch.long),
        col=torch.cat(col) if col else torch.zeros(0, dtype=torch.long),
        value=torch.cat(val) if val else torch.zeros(0),
        sparse_sizes=(dim(Rs_out), dim(Rs_in1) * dim(Rs_in2)))

    return Rs_out, wigner_3j_tensor


class ElementwiseTensorProduct(torch.nn.Module):
    """
    Module for elementwise_tensor_product
    """
    def __init__(self, Rs_in1, Rs_in2, selection_rule=o3.selection_rule):
        super().__init__()

        self.Rs_in1 = simplify(Rs_in1)
        self.Rs_in2 = simplify(Rs_in2)
        assert sum(mul for mul, _, _ in self.Rs_in1) == sum(mul for mul, _, _ in self.Rs_in2)

        self.Rs_out, mixing_matrix = elementwise_tensor_product(self.Rs_in1, self.Rs_in2, selection_rule)
        register_sparse_buffer(self, "mixing_matrix", mixing_matrix)

    def forward(self, features_1, features_2):
        '''
        :param features_1: [..., in1]
        :param features_2: [..., in2]
        :return: [..., out]
        '''
        d_out = dim(self.Rs_out)
        d_in1 = dim(self.Rs_in1)
        d_in2 = dim(self.Rs_in2)

        features = features_1[..., :, None] * features_2[..., None, :]

        size = features.shape[:-2]
        features = features.reshape(-1, d_in1, d_in2)  # [in1, in2, batch]

        mixing_matrix = get_sparse_buffer(self, "mixing_matrix")  # [out, in1 * in2]

        features = torch.einsum('zij->ijz', features)  # [in1, in2, batch]
        features = features.reshape(d_in1 * d_in2, features.shape[2])
        features = mixing_matrix @ features  # [out, batch]
        return features.T.reshape(*size, d_out)


def _is_representation(D, eps, with_parity=False):
    e = (0, 0, 0, 0) if with_parity else (0, 0, 0)
    I = D(*e)
    if not torch.allclose(I, I @ I):
        return False

    g1 = o3.rand_angles() + (0,) if with_parity else o3.rand_angles()
    g2 = o3.rand_angles() + (0,) if with_parity else o3.rand_angles()

    g12 = o3.compose_with_parity(*g1, *g2) if with_parity else o3.compose(*g1, *g2)
    D12 = D(*g12)

    D1D2 = D(*g1) @ D(*g2)

    return (D12 - D1D2).abs().max().item() < eps * D12.abs().max().item()


def _round_sqrt(x, eps):
    x[x.abs() < eps] = 0
    x = x.sign() / x.pow(2)
    x = x.div(eps).round().mul(eps)
    x = x.sign() / x.abs().sqrt()
    x[torch.isnan(x)] = 0
    x[torch.isinf(x)] = 0
    return x


def reduce_tensor(formula, eps=1e-9, has_parity=None, **kw_Rs):
    """
    Usage
    Rs, Q = rs.reduce_tensor('ijkl=jikl=ikjl=ijlk', i=[(1, 1)])
    Rs = 0,2,4
    Q = tensor of shape [15, 81]
    """
    with torch_default_dtype(torch.float64):
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

        formulas = {(s, tuple(f.index(i) for i in f0)) for s, f in formulas}  # set of generators (permutations)

        # create the entire group
        while True:
            n = len(formulas)
            formulas = formulas.union([(s, perm.inverse(p)) for s, p in formulas])
            formulas = formulas.union([
                (s1 * s2, perm.compose(p1, p2))
                for s1, p1 in formulas
                for s2, p2 in formulas
            ])
            if len(formulas) == n:
                break

        for i in kw_Rs:
            if not callable(kw_Rs[i]):
                Rs = convention(kw_Rs[i])
                if has_parity is None:
                    has_parity = any(p != 0 for _, _, p in Rs)
                if not has_parity and not all(p == 0 for _, _, p in Rs):
                    raise RuntimeError(f'{format_Rs(Rs)} parity has to be specified everywhere or nowhere')
                if has_parity and any(p == 0 for _, _, p in Rs):
                    raise RuntimeError(f'{format_Rs(Rs)} parity has to be specified everywhere or nowhere')
                kw_Rs[i] = Rs

        if has_parity is None:
            raise RuntimeError(f'please specify the argument `has_parity`')

        for _s, p in formulas:
            f = "".join(f0[i] for i in p)
            for i, j in zip(f0, f):
                if i in kw_Rs and j in kw_Rs and kw_Rs[i] != kw_Rs[j]:
                    raise RuntimeError(f'Rs of {i} (Rs={format_Rs(kw_Rs[i])}) and {j} (Rs={format_Rs(kw_Rs[j])}) should be the same')
                if i in kw_Rs:
                    kw_Rs[j] = kw_Rs[i]
                if j in kw_Rs:
                    kw_Rs[i] = kw_Rs[j]

        for i in f0:
            if i not in kw_Rs:
                raise RuntimeError(f'index {i} has not Rs associated to it')

        e = (0, 0, 0, 0) if has_parity else (0, 0, 0)
        full_base = list(itertools.product(*(range(len(kw_Rs[i](*e)) if callable(kw_Rs[i]) else dim(kw_Rs[i])) for i in f0)))

        base = set()
        for x in full_base:
            xs = {(s, tuple(x[i] for i in p)) for s, p in formulas}
            # s * T[x] all equal for (s, x) in xs
            if not (-1, x) in xs:
                # the sign is arbitrary, put both possibilities
                base.add(frozenset({
                    frozenset(xs),
                    frozenset({(-s, x) for s, x in xs})
                }))

        base = sorted([sorted([sorted(xs) for xs in x]) for x in base])  # requested for python 3.7 but not for 3.8 (probably a bug in 3.7)

        d_sym = len(base)
        d = len(full_base)
        Q = torch.zeros(d_sym, d)

        for i, x in enumerate(base):
            x = max(x, key=lambda xs: sum(s for s, x in xs))
            for s, e in x:
                j = full_base.index(e)
                Q[i, j] = s / len(x)**0.5

        assert torch.allclose(Q @ Q.T, torch.eye(d_sym))

        if d_sym == 0:
            return [], torch.zeros(d_sym, d)

        def representation(alpha, beta, gamma, parity=None):
            def re(r):
                if callable(r):
                    if has_parity:
                        return r(alpha, beta, gamma, parity)
                    return r(alpha, beta, gamma)
                return rep(r, alpha, beta, gamma, parity)

            m = o3.kron(*(re(kw_Rs[i]) for i in f0))
            return Q @ m @ Q.T

        assert _is_representation(representation, eps, has_parity)

        Rs_out = []
        A = Q.clone()
        for l in range(int((d_sym - 1) // 2) + 1):
            for p in [-1, 1] if has_parity else [0]:
                if 2 * l + 1 > d_sym - dim(Rs_out):
                    break

                mul, B, representation = o3.reduce(representation, partial(rep, [(1, l, p)]), eps, has_parity)
                A = o3.direct_sum(torch.eye(d_sym - B.shape[0]), B) @ A
                A = _round_sqrt(A, eps)
                Rs_out += [(mul, l, p)]

                if dim(Rs_out) == d_sym:
                    break

        if dim(Rs_out) != d_sym:
            raise RuntimeError(f'unable to decompose into irreducible representations')
        return simplify(Rs_out), A
