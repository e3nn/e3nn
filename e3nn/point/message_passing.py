# pylint: disable=arguments-differ, redefined-builtin, missing-docstring, no-member, invalid-name, line-too-long, not-callable
import torch
import torch_geometric

# dev
import math
from itertools import accumulate

from e3nn import rs
from e3nn import tensor_message
from e3nn.o3 import get_flat_coupling_coefficients
from e3nn.rsh import spherical_harmonics_xyz


class Convolution(torch_geometric.nn.MessagePassing):
    def __init__(self, kernel):
        super(Convolution, self).__init__(aggr='add', flow='target_to_source')
        self.kernel = kernel

    def forward(self, features, edge_index, edge_r, size=None, n_norm=1):
        """
        :param features: Tensor of shape [n_target, dim(Rs_in)]
        :param edge_index: LongTensor of shape [2, num_messages]
                           edge_index[0] = sources (convolution centers)
                           edge_index[1] = targets (neighbors)
        :param edge_r: Tensor of shape [num_messages, 3]
                       edge_r = position_target - position_source
        :param size: (n_source, n_target) or None
        :param n_norm: typical number of targets per source

        :return: Tensor of shape [n_source, dim(Rs_out)]
        """
        k = self.kernel(edge_r)
        k.div_(n_norm ** 0.5)
        return self.propagate(edge_index, size=size, x=features, k=k)

    def message(self, x_j, k):
        if k.shape[0] == 0:  # https://github.com/pytorch/pytorch/issues/37628
            return torch.zeros(0, k.shape[1])
        return torch.einsum('eij,ej->ei', k, x_j)


class MinimalNetwork(torch.nn.Module):
    def __init__(self, representations):
        super().__init__()

        from functools import partial

        from e3nn.radial import GaussianRadialModel
        from e3nn.non_linearities.rescaled_act import swish

        self.representations = representations

        max_l_out, max_l_in, max_l = self._find_max_l_out_in()
        self.max_l_out = max_l_out
        self.max_l_in = max_l_in
        self.max_l = max_l

        coupling_coefficients, coupling_coefficients_offsets = get_flat_coupling_coefficients(max_l_out, max_l_in)
        self.register_buffer('coupling_coefficients', coupling_coefficients)
        self.register_buffer('coupling_coefficients_offsets', coupling_coefficients_offsets)

        Rs_in = representations[0]
        Rs_out = representations[-1]

        radial_model = partial(GaussianRadialModel, max_radius=3.2, min_radius=0.7, number_of_basis=10, h=100, L=3, act=swish)

        self.layer = TensorPassingLayer(Rs_in, Rs_out, radial_model, self.coupling_coefficients, self.coupling_coefficients_offsets)
        # self.layer2
        # self.layer3

    def _find_max_l_out_in(self):
        max_l_out = max_l_in = max_l = 0

        for Rs_in, Rs_out in zip(self.representations[:-1], self.representations[1:]):
            # running values are need to make sure max_l is not higher that needed (consequent layers)
            # e.g, rotational orders: [4, 6, 4] -> 12 (without), but [4, 6, 4] -> 10 (with)
            running_max_l_out = max([l_out for (_, l_out, *_) in Rs_out])
            running_max_l_in = max([l_in for (_, l_in, *_) in Rs_in])

            max_l_out = max(running_max_l_out, max_l_out)
            max_l_in = max(running_max_l_in, max_l_in)
            max_l = max(running_max_l_out + running_max_l_in, max_l)

        return max_l_out, max_l_in, max_l

    def forward(self, graph):
        Y = self.real_spherical_harmonics(list(range(self.max_l + 1)), graph.rel_vec).contiguous()
        return self.layer(graph.edge_index, graph.x, graph.edge_attr, Y)


class TensorPassingLayer(torch_geometric.nn.MessagePassing):
    def __init__(self, Rs_in, Rs_out, radial_model, coupling_coefficients, coupling_coefficients_offsets):
        super().__init__()

        self.Rs_in = Rs_in
        self.Rs_out = Rs_out
        self.max_l = rs.lmax(Rs_in) + rs.lmax(Rs_out)

        self.register_buffer('l_out_list',      torch.tensor([l_out for (_, l_out, *_) in Rs_out],  dtype=torch.int32))
        self.register_buffer('l_in_list',       torch.tensor([l_in for (_, l_in, *_) in Rs_in],     dtype=torch.int32))
        self.register_buffer('mul_out_list',    torch.tensor([mul_out for (mul_out, *_) in Rs_out], dtype=torch.int32))
        self.register_buffer('mul_in_list',     torch.tensor([mul_in for (mul_in, *_) in Rs_in],    dtype=torch.int32))

        R_base_offsets, grad_base_offsets, features_base_offsets = self._calculate_offsets()
        self.register_buffer('R_base_offsets',          torch.tensor(R_base_offsets,                dtype=torch.int32))
        self.register_buffer('grad_base_offsets',       torch.tensor(grad_base_offsets,             dtype=torch.int32))
        self.register_buffer('features_base_offsets',   torch.tensor(features_base_offsets,         dtype=torch.int32))

        norm_coef = self._calculate_normalization_coefficients()
        self.register_buffer('norm_coef', norm_coef)

        n_radial_model_outputs = self.R_base_offsets[-1].item()
        self.radial_model = radial_model(n_radial_model_outputs)

        self.register_buffer('coupling_coefficients', coupling_coefficients)
        self.register_buffer('coupling_coefficients_offsets', coupling_coefficients_offsets)

    def _calculate_offsets(self):
        R_base_offsets        = [0] + list(accumulate([mul_out * mul_in * (2 * min(l_out, l_in) + 1) for (mul_out, l_out) in zip(self.mul_out_list, self.l_out_list) for (mul_in, l_in) in zip(self.mul_in_list, self.l_in_list)]))
        grad_base_offsets     = [0] + list(accumulate(mul_out * (2 * l_out + 1) for (mul_out, l_out) in zip(self.mul_out_list, self.l_out_list)))
        features_base_offsets = [0] + list(accumulate(mul_in * (2 * l_in + 1) for (mul_in, l_in) in zip(self.mul_in_list, self.l_in_list)))
        return R_base_offsets, grad_base_offsets, features_base_offsets

    def _calculate_normalization_coefficients(self):
        norm_coef = torch.empty((len(self.l_out_list), len(self.l_in_list)))
        for i, l_out in enumerate(self.l_out_list):
            num_summed_elements = sum([mul_in * (2 * min(l_out, l_in) + 1) for mul_in, l_in in zip(self.mul_in_list, self.l_in_list)])  # (l_out + l_in) - |l_out - l_in| = 2*min(l_out, l_in)
            for j, (mul_in, l_in) in enumerate(zip(self.mul_in_list, self.l_in_list)):
                norm_coef[i, j] = math.sqrt(4 * math.pi) * math.sqrt(2 * l_out + 1) / math.sqrt(num_summed_elements)

        return norm_coef

    def forward(self, edge_index, features, abs_distances, rel_vectors):
        radial_model_outputs = self.radial_model(abs_distances)
        real_spherical_harmonics = spherical_harmonics_xyz(list(range(self.max_l + 1)), rel_vectors)
        return self.propagate(edge_index, x=features, rsh=real_spherical_harmonics, rbm=radial_model_outputs)

    def message(self, x_j, rsh, rbm):
        return TensorMessageFunction.apply(x_j, rsh, rbm, self)


class TensorMessageFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, features, real_spherical_harmonics, radial_model_outputs, layer):
        F = features.transpose(0, 1).contiguous()
        Y = real_spherical_harmonics.transpose(0, 1).contiguous()
        R = radial_model_outputs.transpose(0, 1).contiguous()

        msg = tensor_message.forward(
            layer.norm_coef,                                    # W
            layer.coupling_coefficients,                        # C
            F,                                                  # F
            Y,                                                  # Y
            R,                                                  # R
            layer.l_out_list,                                   # L_out_list
            layer.l_in_list,                                    # L_in_list
            layer.mul_out_list,                                 # u_sizes
            layer.mul_in_list,                                  # v_sizes
            layer.grad_base_offsets,                            # output_base_offsets
            layer.coupling_coefficients_offsets,                # C_offsets
            layer.features_base_offsets,                        # F_base_offsets
            layer.R_base_offsets)                               # R_base_offsets

        msg = msg.transpose_(0, 1).contiguous()                 # [(l_out, u, i), (a, b)] -> [(a, b), (l_out, u, i)]

        if features.requires_grad or radial_model_outputs.requires_grad:
            ctx.save_for_backward(F, Y, R)                      # F, Y, R - transposed
            ctx.layer = layer

        return msg

    @staticmethod
    def backward(ctx, grad_output):
        F, Y, R = ctx.saved_tensors  # F, R - transposed
        layer = ctx.layer

        G = grad_output.transpose(0, 1).contiguous()

        msg_grad_F = tensor_message.backward_F(
            layer.norm_coef,                                    # W
            layer.coupling_coefficients,                        # C
            G,                                                  # G
            Y,                                                  # Y
            R,                                                  # R
            layer.l_out_list,                                   # L_out_list
            layer.l_in_list,                                    # L_in_list
            layer.mul_out_list,                                 # u_sizes
            layer.mul_in_list,                                  # v_sizes
            layer.features_base_offsets,                        # output_base_offsets
            layer.coupling_coefficients_offsets,                # C_offsets
            layer.grad_base_offsets,                            # G_base_offsets
            layer.R_base_offsets)                               # R_base_offsets

        msg_grad_F = msg_grad_F.transpose_(0, 1).contiguous()   # [(a, b), (l_in, v, j)]

        msg_grad_R = tensor_message.backward_R(
            layer.norm_coef,                                    # W
            layer.coupling_coefficients,                        # C
            G,                                                  # G
            F,                                                  # F
            Y,                                                  # Y
            layer.l_out_list,                                   # L_out_list
            layer.l_in_list,                                    # L_in_list
            layer.mul_out_list,                                 # u_sizes
            layer.mul_in_list,                                  # v_sizes
            layer.R_base_offsets,                               # output_base_offsets
            layer.coupling_coefficients_offsets,                # C_offsets
            layer.grad_base_offsets,                            # G_base_offsets
            layer.features_base_offsets)                        # F_base_offsets

        msg_grad_R = msg_grad_R.transpose_(0, 1).contiguous()   # [(a, b), (l_out, l_in, l, u, v)]

        return msg_grad_F, None, msg_grad_R, None
