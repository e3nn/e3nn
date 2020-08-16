# pylint: disable=invalid-name, arguments-differ, missing-docstring, line-too-long, no-member, redefined-builtin, abstract-method
from functools import partial

import torch
from e3nn import o3, rs
from e3nn.linear_mod import KernelLinear
from e3nn.util.eval_code import eval_code
from e3nn.util.sparse import get_sparse_buffer, register_sparse_buffer


class LearnableTensorSquare(torch.nn.Module):
    def __init__(self, Rs_in, Rs_out, linear=True, allow_change_output=False, allow_zero_outputs=False):
        super().__init__()

        self.Rs_in = rs.simplify(Rs_in)
        self.Rs_out = rs.simplify(Rs_out)

        ls = [l for _, l, _ in self.Rs_out]
        selection_rule = partial(o3.selection_rule, lfilter=lambda l: l in ls)

        if linear:
            Rs_in = [(1, 0, 1)] + self.Rs_in
        else:
            Rs_in = self.Rs_in
        self.linear = linear

        Rs_ts, T = rs.tensor_square(Rs_in, selection_rule)
        register_sparse_buffer(self, 'T', T)  # [out, in1 * in2]

        ls = [l for _, l, _ in Rs_ts]
        if allow_change_output:
            self.Rs_out = [(mul, l, p) for mul, l, p in self.Rs_out if l in ls]
        elif not allow_zero_outputs:
            assert all(l in ls for _, l, _ in self.Rs_out)

        self.kernel = KernelLinear(Rs_ts, self.Rs_out)  # [out, in, w]

    def __repr__(self):
        return "{name} ({Rs_in} -> {Rs_out})".format(
            name=self.__class__.__name__,
            Rs_in=rs.format_Rs(self.Rs_in),
            Rs_out=rs.format_Rs(self.Rs_out),
        )

    def forward(self, features):
        '''
        :param features: [..., channels]
        '''
        *size, n = features.size()
        features = features.reshape(-1, n)
        assert n == rs.dim(self.Rs_in)

        if self.linear:
            features = torch.cat([features.new_ones(features.shape[0], 1), features], dim=1)
            n += 1

        T = get_sparse_buffer(self, 'T')  # [out, in1 * in2]
        kernel = (T.t() @ self.kernel().T).T.reshape(rs.dim(self.Rs_out), n, n)  # [out, in1, in2]
        features = torch.einsum('zi,zj->zij', features, features)
        features = torch.einsum('kij,zij->zk', kernel, features)
        return features.reshape(*size, -1)


class LearnableTensorProduct(torch.nn.Module):
    def __init__(self, Rs_in1, Rs_in2, Rs_out, allow_change_output=False):
        super().__init__()

        self.Rs_in1 = rs.simplify(Rs_in1)
        self.Rs_in2 = rs.simplify(Rs_in2)
        self.Rs_out = rs.simplify(Rs_out)

        ls = [l for _, l, _ in self.Rs_out]
        selection_rule = partial(o3.selection_rule, lfilter=lambda l: l in ls)

        Rs_ts, T = rs.tensor_product(self.Rs_in1, self.Rs_in2, selection_rule)
        register_sparse_buffer(self, 'T', T)  # [out, in1 * in2]

        ls = [l for _, l, _ in Rs_ts]
        if allow_change_output:
            self.Rs_out = [(mul, l, p) for mul, l, p in self.Rs_out if l in ls]
        else:
            assert all(l in ls for _, l, _ in self.Rs_out)

        self.kernel = KernelLinear(Rs_ts, self.Rs_out)  # [out, in, w]

    def forward(self, features_1, features_2):
        """
        :return:         tensor [..., channel]
        """
        *size, n = features_1.size()
        features_1 = features_1.reshape(-1, n)
        assert n == rs.dim(self.Rs_in1)
        *size2, n = features_2.size()
        features_2 = features_2.reshape(-1, n)
        assert size == size2

        T = get_sparse_buffer(self, 'T')  # [out, in1 * in2]
        kernel = (T.t() @ self.kernel().T).T.reshape(rs.dim(self.Rs_out), rs.dim(self.Rs_in1), rs.dim(self.Rs_in2))  # [out, in1, in2]
        features = torch.einsum('kij,zi,zj->zk', kernel, features_1, features_2)
        return features.reshape(*size, -1)


_tensor_product_code = """
import torch

@torch.jit.script
def main(ARGS, x1: torch.Tensor, x2: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    batch = x1.shape[0]
    out = x1.new_zeros((batch, DIM))
    ein = torch.einsum

CODE
    return out
"""


class WeightedTensorProduct(torch.nn.Module):
    def __init__(self, Rs_in1, Rs_in2, Rs_out, selection_rule=o3.selection_rule, normalization='component', groups=1):
        super().__init__()

        self.Rs_in1 = rs.convention(Rs_in1)
        self.Rs_in2 = rs.convention(Rs_in2)
        self.Rs_out = rs.convention(Rs_out)

        code = ""

        index_w = 0
        wigners = set()
        count = [0 for _ in range(rs.dim(self.Rs_out))]

        index_1 = 0
        for mul_1, l_1, p_1 in self.Rs_in1:
            dim_1 = mul_1 * (2 * l_1 + 1)

            index_2 = 0
            for mul_2, l_2, p_2 in self.Rs_in2:
                dim_2 = mul_2 * (2 * l_2 + 1)

                gmul_1s = [mul_1 // groups + (g < mul_1 % groups) for g in range(groups)]
                gmul_2s = [mul_2 // groups + (g < mul_2 % groups) for g in range(groups)]

                for g in range(groups):
                    if gmul_1s[g] * gmul_2s[g] == 0:
                        continue

                    code += f"    s1 = x1[:, {index_1+sum(gmul_1s[:g])*(2*l_1+1)}:{index_1+sum(gmul_1s[:g+1])*(2*l_1+1)}].reshape(batch, {gmul_1s[g]}, {2 * l_1 + 1})\n"
                    code += f"    s2 = x2[:, {index_2+sum(gmul_2s[:g])*(2*l_2+1)}:{index_2+sum(gmul_2s[:g+1])*(2*l_2+1)}].reshape(batch, {gmul_2s[g]}, {2 * l_2 + 1})\n"
                    code += f"    ss = ein('zui,zvj->zuvij', s1, s2)\n"

                    index_out = 0
                    for mul_out, l_out, p_out in self.Rs_out:
                        dim_out = mul_out * (2 * l_out + 1)

                        if l_out in selection_rule(l_1, p_1, l_2, p_2) and p_out == p_1 * p_2:
                            wigners.add((l_out, l_1, l_2))

                            gmul_outs = [mul_out // groups + (g < mul_out % groups) for g in range(groups)]
                            dim_w = gmul_outs[g] * gmul_1s[g] * gmul_2s[g]

                            if gmul_outs[g] == 0:
                                continue

                            code += f"    sw = w[:, {index_w}:{index_w+dim_w}].reshape(batch, {gmul_outs[g]}, {gmul_1s[g]}, {gmul_2s[g]})\n"
                            i = index_out+sum(gmul_outs[:g])*(2*l_out+1)
                            j = index_out+sum(gmul_outs[:g+1])*(2*l_out+1)
                            code += f"    out[:, {i}:{j}] += ein('zwuv,kij,zuvij->zwk', sw, C{l_out}_{l_1}_{l_2}, ss).reshape(batch, {gmul_outs[g]*(2*l_out+1)})\n"
                            code += "\n"

                            for k in range(i, j):
                                count[k] += gmul_1s[g] * gmul_2s[g]

                            index_w += dim_w

                        index_out += dim_out

                index_2 += dim_2
            index_1 += dim_1

        ilast = 0
        clast = count[0]
        for i, c in enumerate(count):
            if clast != c:
                if clast > 1:
                    code += f"    out[:, {ilast}:{i}].div_({clast ** 0.5})\n"
                clast = c
                ilast = i
        if clast > 1:
            code += f"    out[:, {ilast}:].div_({clast ** 0.5})\n"

        wigners = sorted(wigners)
        self.wigners_names = [f"C{l_out}_{l_1}_{l_2}" for l_out, l_1, l_2 in wigners]
        args = ", ".join(f"{arg}: torch.Tensor" for arg in self.wigners_names)

        for arg, (l_out, l_1, l_2) in zip(self.wigners_names, wigners):
            C = o3.wigner_3j(l_out, l_1, l_2)

            if normalization == 'component':
                C *= (2 * l_out + 1) ** 0.5
            if normalization == 'norm':
                C *= (2 * l_1 + 1) ** 0.5 * (2 * l_2 + 1) ** 0.5

            self.register_buffer(arg, C)

        x = _tensor_product_code
        x = x.replace("DIM", f"{rs.dim(self.Rs_out)}")
        x = x.replace("ARGS", args)
        x = x.replace("CODE", code)

        self.main = eval_code(x).main
        self.nweight = index_w

    def forward(self, features_1, features_2, weights):
        """
        :return:         tensor [..., channel]
        """
        *size, n = features_1.size()
        features_1 = features_1.reshape(-1, n)
        assert n == rs.dim(self.Rs_in1), f"{n} is not {rs.dim(self.Rs_in1)}"
        *size2, n = features_2.size()
        features_2 = features_2.reshape(-1, n)
        assert n == rs.dim(self.Rs_in2), f"{n} is not {rs.dim(self.Rs_in2)}"
        assert size == size2
        weights = weights.reshape(-1, self.nweight)

        wigners = [getattr(self, arg) for arg in self.wigners_names]

        features = self.main(*wigners, features_1, features_2, weights)
        return features.reshape(*size, -1)
