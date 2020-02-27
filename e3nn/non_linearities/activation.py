# pylint: disable=invalid-name, arguments-differ, missing-docstring, no-member
import copy

import torch

from e3nn import rs


class Activation(torch.nn.Module):
    def __init__(self, Rs, acts):
        '''
        Can be used only with scalar fields

        :param acts: list of tuple (multiplicity, activation)
        '''
        super().__init__()

        Rs = rs.simplify(Rs)
        acts = copy.deepcopy(acts)

        n1 = sum(mul for mul, _, _ in Rs)
        n2 = sum(mul for mul, _ in acts if mul > 0)

        for i, (mul, act) in enumerate(acts):
            if mul == -1:
                acts[i] = (n1 - n2, act)
                assert n1 - n2 >= 0

        assert n1 == sum(mul for mul, _ in acts)

        i = 0
        while i < len(Rs):
            mul_r, l, p_r = Rs[i]
            mul_a, act = acts[i]

            if mul_r < mul_a:
                acts[i] = (mul_r, act)
                acts.insert(i + 1, (mul_a - mul_r, act))

            if mul_a < mul_r:
                Rs[i] = (mul_a, l, p_r)
                Rs.insert(i + 1, (mul_r - mul_a, l, p_r))
            i += 1

        x = torch.linspace(0, 10, 256)

        Rs_out = []
        for (mul, l, p_in), (mul_a, act) in zip(Rs, acts):
            assert mul == mul_a

            a1, a2 = act(x), act(-x)
            if (a1 - a2).abs().max() < a1.abs().max() * 1e-10:
                p_act = 1
            elif (a1 + a2).abs().max() < a1.abs().max() * 1e-10:
                p_act = -1
            else:
                p_act = 0

            p = p_act if p_in == -1 else p_in
            Rs_out.append((mul, 0, p))

            if p_in != 0 and p == 0:
                raise ValueError("warning! the parity is violated")

        self.Rs_out = Rs_out
        self.acts = acts

    def forward(self, features, dim=-1):
        '''
        :param features: [..., channels, ...]
        '''
        output = []
        index = 0
        for mul, act in self.acts:
            output.append(act(features.narrow(dim, index, mul)))
            index += mul

        if output:
            return torch.cat(output, dim=dim)
        else:
            size = list(features.size())
            size[dim] = 0
            return features.new_zeros(*size)
