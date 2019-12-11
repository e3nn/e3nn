# pylint: disable=invalid-name, arguments-differ, missing-docstring, line-too-long, no-member
import torch

from e3nn import SO3


class Multiplication(torch.nn.Module):
    def __init__(self, Rs_1, Rs_2):
        super().__init__()

        Rs_1 = SO3.normalizeRs(Rs_1)
        Rs_2 = SO3.normalizeRs(Rs_2)
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

        self.Rs_1 = Rs_1
        self.Rs_2 = Rs_2

        Rs_out = []
        for (mul, l_1, p_1), (mul_2, l_2, p_2) in zip(Rs_1, Rs_2):
            assert mul == mul_2
            for l in range(abs(l_1 - l_2), l_1 + l_2 + 1):
                Rs_out.append((mul, l, p_1 * p_2))

                C = SO3.clebsch_gordan(l, l_1, l_2).type(torch.get_default_dtype()) * (2 * l + 1) ** 0.5
                if l_1 == 0 or l_2 == 0:
                    m = C.view(2 * l + 1, 2 * l + 1)
                    if C.dtype == torch.float:
                        assert (m - torch.eye(2 * l + 1, dtype=C.dtype)).abs().max() < 1e-7, m.numpy().round(3)
                    else:
                        assert (m - torch.eye(2 * l + 1, dtype=C.dtype)).abs().max() < 1e-10, m.numpy().round(3)
                else:
                    self.register_buffer("cg_{}_{}_{}".format(l, l_1, l_2), C)

        self.Rs_out = Rs_out


    def forward(self, features_1, features_2):
        '''
        :param features: [..., channels]
        '''
        *size_1, n_1 = features_1.size()
        features_1 = features_1.view(-1, n_1)
        *size_2, n_2 = features_2.size()
        features_2 = features_2.view(-1, n_2)
        assert size_1 == size_2
        batch = features_1.size(0)

        output = []
        index_1 = 0
        index_2 = 0
        for (mul, l_1, _), (_, l_2, _) in zip(self.Rs_1, self.Rs_2):
            f_1 = features_1.narrow(1, index_1, mul * (2 * l_1 + 1)).reshape(batch * mul, 2 * l_1 + 1)  # [z, j]
            index_1 += mul * (2 * l_1 + 1)

            f_2 = features_2.narrow(1, index_2, mul * (2 * l_2 + 1)).reshape(batch * mul, 2 * l_2 + 1)  # [z, k]
            index_2 += mul * (2 * l_2 + 1)

            if l_1 == 0 or l_2 == 0:
                # special case optimization
                out = f_1 * f_2
                output.append(out.view(batch, mul * (2 * max(l_1, l_2) + 1)))
            else:
                for l in range(abs(l_1 - l_2), l_1 + l_2 + 1):
                    C = getattr(self, "cg_{}_{}_{}".format(l, l_1, l_2))
                    out = torch.einsum("ijk,zj,zk->zi", (C, f_1, f_2)).view(batch, mul * (2 * l + 1))  # [z, i]
                    output.append(out.view(batch, mul * (2 * l + 1)))
        assert index_1 == features_1.size(1)
        assert index_2 == features_2.size(1)

        output = torch.cat(output, dim=1)
        return output.view(*size_1, -1)
