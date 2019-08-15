# pylint: disable=invalid-name, arguments-differ, missing-docstring, no-member
import torch

from se3cnn import SO3


class Activation(torch.nn.Module):
    def __init__(self, Rs, acts):
        '''
        Can be used only with scalar fields

        :param acts: list of tuple (multiplicity, activation, activation parity)
        '''
        super().__init__()

        Rs = SO3.normalizeRs(Rs)
        assert sum(mul for mul, _, _ in Rs) == sum(mul for mul, _, _ in acts)

        i = 0
        while i < len(Rs):
            mul_r, l, p_r = Rs[i]
            mul_a, act, p_a = acts[i]

            if mul_r < mul_a:
                acts[i] = (mul_r, act, p_a)
                acts.insert(i + 1, (mul_a - mul_r, act, p_a))

            if mul_a < mul_r:
                Rs[i] = (mul_a, l, p_r)
                Rs.insert(i + 1, (mul_r - mul_a, l, p_r))
            i += 1

        Rs_out = []
        for (mul, l, p_in), (mul_a, _act, p_act) in zip(Rs, acts):
            assert mul == mul_a
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
        for mul, act, _ in self.acts:
            output.append(act(features.narrow(dim, index, mul)))
            index += mul

        if output:
            return torch.cat(output, dim=dim)
        else:
            size = list(features.size())
            size[dim] = 0
            return features.new_zeros(*size)
