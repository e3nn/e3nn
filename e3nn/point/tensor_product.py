import torch
import e3nn
import e3nn.SO3 as SO3
import e3nn.util.manipulations as manip
import e3nn.point.self_interaction as si

torch.set_default_dtype(torch.float64)

class LearnablePathTensorProduct(torch.nn.Module):
    def __init__(self, Rs, L_max):
        super().__init__()
        muls, Ls = zip(*Rs)
        if len(set(muls)) != 1:
            raise ValueError("All Ls must have the same mulitplicity.")
        mul = list(set(muls))[0]
        Rs_single = [(1, L) for L in range(L_max + 1)]
        Rs_new_trunc, Q_trunc = manip.get_truncated_shuffled_Q(Rs_single)
        rearrange, indices = manip.split_Rs(Rs)
        self.register_buffer('Q_trunc', Q_trunc)
        self.register_buffer('rearrange', rearrange)
        self.Rs_new_trunc = Rs_new_trunc
        self.si = si.SelfInteraction(Rs_new_trunc * mul, Rs)

    def forward(self, input):
        input_rearrange = torch.einsum('mci,zai->zamc', self.rearrange, input)
        tensor_product = torch.einsum('cdf,zamc,zamd->zamf', self.Q_trunc, input_rearrange, input_rearrange)
        shape = tensor_product.shape
        shape = list(shape[:-2]) + [-1]
        return self.si(tensor_product.view(shape))

