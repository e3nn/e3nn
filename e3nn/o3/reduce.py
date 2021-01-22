r"""Function to decompose a multi-index tensor
"""
import collections
import torch
from e3nn import o3
from e3nn.math import group, orthonormalize


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


_TP = collections.namedtuple('tp', 'op, args')
_INPUT = collections.namedtuple('input', 'tensor, start, stop')


def _wigner_nj(*irrepss, normalization='component'):
    irrepss = [o3.Irreps(irreps) for irreps in irrepss]

    if len(irrepss) == 1:
        irreps, = irrepss
        ret = []
        e = torch.eye(irreps.dim)
        i = 0
        for mul, ir in irreps:
            for _ in range(mul):
                sl = slice(i, i + ir.dim)
                ret += [
                    (ir, _INPUT(0, sl.start, sl.stop), e[sl])
                ]
                i += ir.dim
        return ret

    *irrepss_left, irreps_right = irrepss
    ret = []
    for ir_left, path_left, C_left in _wigner_nj(*irrepss_left):
        i = 0
        for mul, ir in irreps_right:
            for ir_out in ir_left * ir:
                C = o3.wigner_3j(ir_out.l, ir_left.l, ir.l)
                if normalization == 'component':
                    C *= ir_out.dim**0.5
                if normalization == 'norm':
                    C *= ir_left.dim**0.5 * ir.dim**0.5

                C = torch.einsum('jk,ijl->ikl', C_left.flatten(1), C)
                C = C.reshape(ir_out.dim, *(irreps.dim for irreps in irrepss_left), ir.dim)
                for u in range(mul):
                    E = torch.zeros(ir_out.dim, *(irreps.dim for irreps in irrepss_left), irreps_right.dim)
                    sl = slice(i + u * ir.dim, i + (u+1) * ir.dim)
                    E[..., sl] = C
                    ret += [
                        (
                            ir_out,
                            _TP(
                                op=(ir_left, ir, ir_out),
                                args=(path_left, _INPUT(len(irrepss_left), sl.start, sl.stop))
                            ),
                            E
                        )
                    ]
            i += mul * ir.dim

    return sorted(ret, key=lambda x: x[0])


class ReducedTensorProducts:
    def __init__(self, formula, ir_out=None, eps=1e-9, **irreps):
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

        Ps = collections.defaultdict(list)

        for ir, path, C in _wigner_nj(*[irreps[i] for i in f0]):
            P = C.flatten(1) @ Q.flatten(1).T
            Ps[ir].append((P.flatten(), path, C))

        tps = set()

        def get_ops(path):
            if isinstance(path, _INPUT):
                return
            assert isinstance(path, _TP)
            yield path.op
            for op in get_ops(path.args[0]):
                yield op

        outputs = []
        change_of_basis = []

        for ir in Ps:
            if ir_out is not None and ir not in ir_out:
                continue

            P = torch.stack([P for P, _, _ in Ps[ir]])
            paths = [path for _, path, _ in Ps[ir]]
            Cs = [C for _, _, C in Ps[ir]]

            P, A = orthonormalize(P, eps)
            P = P.reshape(len(P), ir.dim, len(Q))

            # remove paths
            paths = [path for path, c in zip(paths, A.norm(dim=0)) if c > eps]
            A = A[:, A.norm(dim=0) > eps]
            Cs = [C for C, c in zip(Cs, A.norm(dim=0)) if c > eps]

            assert A.shape[0] == A.shape[1]

            for path in paths:
                for op in get_ops(path):
                    tps.add(op)

            for path in paths:
                outputs.append((
                    ir, path
                ))

            for C in Cs:
                change_of_basis.append(C)

        self.outputs = outputs
        self.change_of_basis = torch.cat(change_of_basis)

        self.tps = {
            op: o3.TensorProduct(op[0], op[1], op[2], [(0, 0, 0, 'uuu', False)])
            for op in tps
        }
        self.irreps_out = o3.Irreps([ir for ir, _ in outputs]).simplify()

    def __call__(self, *xs):
        values = dict()

        def evaluate(path):
            if path in values:
                return values[path]

            if isinstance(path, _INPUT):
                out = xs[path.tensor][..., path.start:path.stop]
                values[path] = out
                return out
            if isinstance(path, _TP):
                out = self.tps[path.op](evaluate(path.args[0]), evaluate(path.args[1]))
                values[path] = out
                return out

        return torch.cat([evaluate(path) for ir, path in self.outputs], dim=-1)
