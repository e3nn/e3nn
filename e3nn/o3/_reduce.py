import collections
import torch
from torch import fx
from e3nn import o3
from e3nn.math import group
from e3nn.util import explicit_default_types
from e3nn.util.jit import compile_mode


_TP = collections.namedtuple('tp', 'op, args')
_INPUT = collections.namedtuple('input', 'tensor, start, stop')


def _wigner_nj(*irrepss, normalization='component', filter_ir_mid=None, dtype=None, device=None):
    irrepss = [o3.Irreps(irreps) for irreps in irrepss]
    if filter_ir_mid is not None:
        filter_ir_mid = [o3.Irrep(ir) for ir in filter_ir_mid]

    if len(irrepss) == 1:
        irreps, = irrepss
        ret = []
        e = torch.eye(irreps.dim, dtype=dtype, device=device)
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
    for ir_left, path_left, C_left in _wigner_nj(*irrepss_left, dtype=dtype, device=device):
        i = 0
        for mul, ir in irreps_right:
            for ir_out in ir_left * ir:
                if filter_ir_mid is not None and ir_out not in filter_ir_mid:
                    continue

                C = o3.wigner_3j(ir_out.l, ir_left.l, ir.l, dtype=dtype, device=device)
                if normalization == 'component':
                    C *= ir_out.dim**0.5
                if normalization == 'norm':
                    C *= ir_left.dim**0.5 * ir.dim**0.5

                C = torch.einsum('jk,ijl->ikl', C_left.flatten(1), C)
                C = C.reshape(ir_out.dim, *(irreps.dim for irreps in irrepss_left), ir.dim)
                for u in range(mul):
                    E = torch.zeros(ir_out.dim, *(irreps.dim for irreps in irrepss_left), irreps_right.dim, dtype=dtype, device=device)
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


def _get_ops(path):
    if isinstance(path, _INPUT):
        return
    assert isinstance(path, _TP)
    yield path.op
    for op in _get_ops(path.args[0]):
        yield op


@compile_mode('script')
class ReducedTensorProducts(fx.GraphModule):
    r"""reduce a tensor with symmetries into irreducible representations

    Parameters
    ----------
    formula : str
        String made of letters ``-`` and ``=`` that represent the indices symmetries of the tensor.
        For instance ``ij=ji`` means that the tensor has to indices and if they are exchanged, its value is the same.
        ``ij=-ji`` means that the tensor change its sign if the two indices are exchanged.

    filter_ir_out : list of `Irrep`, optional
        Optional, list of allowed irrep in the output

    filter_ir_mid : list of `Irrep`, optional
        Optional, list of allowed irrep in the intermediary operations

    **kwargs : dict of `Irreps`
        each letter present in the formula has to be present in the ``irreps`` dictionary, unless it can be inferred by the formula.
        For instance if the formula is ``ij=ji`` you can provide the representation of ``i`` only: ``ReducedTensorProducts('ij=ji', i='1o')``.

    Attributes
    ----------
    irreps_in : list of `Irreps`
        input representations

    irreps_out : `Irreps`
        output representation

    change_of_basis : `torch.Tensor`
        tensor of shape ``(irreps_out.dim, irreps_in[0].dim, ..., irreps_in[-1].dim)``

    Examples
    --------
    >>> tp = ReducedTensorProducts('ij=-ji', i='1o')
    >>> x = torch.tensor([1.0, 0.0, 0.0])
    >>> y = torch.tensor([0.0, 1.0, 0.0])
    >>> tp(x, y) + tp(y, x)
    tensor([0., 0., 0.])

    >>> tp = ReducedTensorProducts('ijkl=jikl=ikjl=ijlk', i="1e")
    >>> tp.irreps_out
    1x0e+1x2e+1x4e

    >>> tp = ReducedTensorProducts('ij=ji', i='1o')
    >>> x, y = torch.randn(2, 3)
    >>> a = torch.einsum('zij,i,j->z', tp.change_of_basis, x, y)
    >>> b = tp(x, y)
    >>> assert torch.allclose(a, b, atol=1e-3, rtol=1e-3)
    """
    def __init__(self, formula, filter_ir_out=None, filter_ir_mid=None, eps=1e-9, **irreps):
        super().__init__(self, fx.Graph())

        if filter_ir_out is not None:
            filter_ir_out = [o3.Irrep(ir) for ir in filter_ir_out]

        f0, formulas = group.germinate_formulas(formula)

        irreps = {i: o3.Irreps(irs) for i, irs in irreps.items()}

        for i in irreps:
            if len(i) != 1:
                raise TypeError(f"got an unexpected keyword argument '{i}'")

        for _sign, p in formulas:
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

        for i in irreps:
            if i not in f0:
                raise RuntimeError(f'index {i} has an irreps but does not appear in the fomula')

        base_perm, _ = group.reduce_permutation(
            f0,
            formulas,
            dtype=torch.float64,
            **{i: irs.dim for i, irs in irreps.items()}
        )

        Ps = collections.defaultdict(list)

        for ir, path, base_o3 in _wigner_nj(*[irreps[i] for i in f0], filter_ir_mid=filter_ir_mid, dtype=torch.float64):
            if filter_ir_out is None or ir in filter_ir_out:
                P = base_o3.flatten(1) @ base_perm.flatten(1).T
                if P.norm() > eps:  # if this Irrep is present in the premutation basis we keep it
                    Ps[ir].append((path, base_o3))

        outputs = []
        change_of_basis = []
        irreps_out = []

        for ir in Ps:
            mul = len(Ps[ir])
            paths = [path for path, _ in Ps[ir]]
            base_o3 = torch.stack([R for _, R in Ps[ir]])

            R = base_o3.flatten(2)  # [multiplicity, ir, input basis] (u,j,omega)
            P = base_perm.flatten(1)  # [permutation basis, input basis] (a,omega)

            Xs = []
            for j in range(ir.dim):
                RR = R[:, j] @ R[:, j].T  # (u,u)
                PP = P @ P.T  # (a,a)
                RP = R[:, j] @ P.T  # (u,a)

                prob = torch.cat([
                    torch.cat([RR, -RP], dim=1),
                    torch.cat([-RP.T, PP], dim=1)
                ], dim=0)
                eigenvalues, eigenvectors = torch.symeig(prob, eigenvectors=True)
                X = eigenvectors[:, eigenvalues < eps][:mul].T  # [solutions, multiplicity]
                X = torch.linalg.qr(X, mode='r').R
                for i, x in enumerate(X):
                    for j in range(i, mul):
                        if x[j] < eps:
                            x.neg_()
                        if x[j] > eps:
                            break

                Xs.append(X)

            for X in Xs:
                assert (X - Xs[0]).abs().max() < eps

            X = Xs[0]
            for x in X:
                C = torch.einsum("u,ui...->i...", x, base_o3)
                correction = (ir.dim / C.pow(2).sum())**0.5
                C = correction * C

                outputs.append([((correction * v).item(), p) for v, p in zip(x, paths) if v.abs() > eps])
                change_of_basis.append(C)
                irreps_out.append((1, ir))

        dtype, _ = explicit_default_types(None, None)
        self.register_buffer('change_of_basis', torch.cat(change_of_basis).to(dtype=dtype))

        tps = set()
        for vp_list in outputs:
            for v, p in vp_list:
                for op in _get_ops(p):
                    tps.add(op)

        tps = list(tps)
        for i, op in enumerate(tps):
            tp = o3.TensorProduct(op[0], op[1], op[2], [(0, 0, 0, 'uuu', False)])
            setattr(self, f'tp{i}', tp)

        graph = fx.Graph()
        inputs = [
            fx.Proxy(graph.placeholder(f"x{i}", torch.Tensor))
            for i in f0
        ]

        self.irreps_in = [irreps[i] for i in f0]
        self.irreps_out = o3.Irreps(irreps_out).simplify()

        values = dict()

        def evaluate(path):
            if path in values:
                return values[path]

            if isinstance(path, _INPUT):
                out = inputs[path.tensor]
                if (path.start, path.stop) != (0, self.irreps_in[path.tensor].dim):
                    out = out.narrow(-1, path.start, path.stop - path.start)
            if isinstance(path, _TP):
                x1 = evaluate(path.args[0]).node
                x2 = evaluate(path.args[1]).node
                out = fx.Proxy(graph.call_module(f'tp{tps.index(path.op)}', (x1, x2)))
            values[path] = out
            return out

        outs = []
        for vp_list in outputs:
            v, p = vp_list[0]
            out = evaluate(p)
            if abs(v - 1.0) > eps:
                out = v * out
            for v, p in vp_list[1:]:
                t = evaluate(p)
                if abs(v - 1.0) > eps:
                    t = v * t
                out = out + t
            outs.append(out)

        out = torch.cat(outs, dim=-1)
        graph.output(out.node)

        self.graph = graph
        self.recompile()

    def __repr__(self):
        return (
            f"ReducedTensorProducts(\n"
            f"    in: {' times '.join(map(repr, self.irreps_in))}\n"
            f"    out: {self.irreps_out}\n"
            ")"
        )
