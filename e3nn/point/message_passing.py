# pylint: disable=arguments-differ, redefined-builtin, missing-docstring, no-member, invalid-name, line-too-long, not-callable
import torch
import torch_geometric

from e3nn import tensor_message


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
    def __init__(self, representations, device):
        super().__init__()

        from functools import partial

        from e3nn.radial import GaussianRadialModel
        from e3nn.non_linearities.rescaled_act import swish
        from e3nn.rsh import spherical_harmonics_xyz_cuda as rsh

        assert len(representations) == 2, "Only minimal networks here"

        self.device = device

        self.representations = representations
        self._find_max_l_out_in()
        self._calculate_coupling_coefficients_and_offsets()

        Rs_in = representations[0]
        Rs_out = representations[-1]

        radial_model = partial(GaussianRadialModel, max_radius=3.2, min_radius=0.7, number_of_basis=10, h=100, L=3, act=swish)

        self.real_spherical_harmonics = partial(rsh, Rs=representations)
        self.layer = TensorPassingLayer(Rs_in, Rs_out, radial_model, self)

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

        assert max_l <= 10, "l > 10 is not supported"
        self.max_l_out = max_l_out
        self.max_l_in = max_l_in
        self.max_l = max_l

    def _calculate_coupling_coefficients_and_offsets(self):
        from e3nn.o3 import wigner_3j

        tmp_base_offsets_list = []
        tmp_position = 0
        for l_out in range(self.max_l_out + 1):
            for l_in in range(self.max_l_in + 1):
                tmp_base_offsets_list.append(tmp_position)
                for l_f in range(abs(l_out - l_in), l_out + l_in + 1):
                    tmp_position += (2 * l_out + 1) * (2 * l_in + 1) * (2 * l_f + 1)

        self.coupling_coefficients = torch.zeros(tmp_position, device=self.device)      # last tmp_cg_pos (not stored in offsets) = size

        for l_out in range(self.max_l_out + 1):
            for l_in in range(self.max_l_in + 1):
                tmp_base_pos = tmp_base_offsets_list[l_out * (self.max_l_in + 1) + l_in]
                for l_f in range(abs(l_out - l_in), l_out + l_in + 1):
                    tmp_flat = wigner_3j(l_out, l_in, l_f).contiguous().view(-1).type(torch.get_default_dtype()).to(self.device)
                    self.coupling_coefficients[tmp_base_pos:tmp_base_pos + tmp_flat.shape[0]] = tmp_flat
                    tmp_base_pos += tmp_flat.shape[0]

        self.coupling_coefficients_base_offsets = torch.tensor(tmp_base_offsets_list, dtype=torch.int32, device=self.device)

    def forward(self, graph):
        return self.layer(graph.edge_index, graph.x, graph.edge_attr, self.real_spherical_harmonics(self.rel_vec))


class TensorPassingLayer(torch_geometric.nn.MessagePassing):
    def __init__(self, Rs_in, Rs_out, radial_model, network):
        super().__init__()

        self.device = network.device

        self.Rs_in = Rs_in
        self.Rs_out = Rs_out

        self.l_out_list = torch.tensor([l_out for (_, l_out, *_) in Rs_out], dtype=torch.int32, device=self.device)
        self.l_in_list = torch.tensor([l_in for (_, l_in, *_) in Rs_in], dtype=torch.int32, device=self.device)
        self.mul_out_list = torch.tensor([mul_out for (mul_out, *_) in Rs_out], dtype=torch.int32, device=self.device)
        self.mul_in_list = torch.tensor([mul_in for (mul_in, *_) in Rs_in], dtype=torch.int32, device=self.device)

        self._calculate_offsets()
        self._calculate_normalization_coefficients()

        self.radial_model = radial_model
        self.network = network

    def _calculate_offsets(self):
        from itertools import accumulate

        R_base_offset = list(accumulate([mul_out * mul_in * (2 * min(l_out, l_in) + 1) for (mul_out, l_out) in zip(self.mul_out_list, self.l_out_list) for (mul_in, l_in) in zip(self.mul_in_list, self.l_in_list)]))
        grad_base_offset = list(accumulate(mul_out * (2 * l_out + 1) for (mul_out, l_out) in zip(self.mul_out_list, self.l_out_list)))
        features_base_offset = list(accumulate(mul_in * (2 * l_in + 1) for (mul_in, l_in) in zip(self.mul_in_list, self.l_in_list)))

        R_base_offset.insert(0, 0)
        grad_base_offset.insert(0, 0)
        features_base_offset.insert(0, 0)

        self.R_base_offsets = torch.tensor(R_base_offset, dtype=torch.int32, device=self.device)
        self.grad_base_offsets = torch.tensor(grad_base_offset, dtype=torch.int32, device=self.device)
        self.features_base_offsets = torch.tensor(features_base_offset, dtype=torch.int32, device=self.device)

    def _calculate_normalization_coefficients(self):
        import math

        norm_coef = torch.zeros((len(self.l_out_list), len(self.l_in_list)), device=self.device)
        for i, l_out in enumerate(self.l_out_list):
            num_summed_elements = sum([mul_in * (2 * min(l_out, l_in) + 1) for mul_in, l_in in zip(self.mul_in_list, self.l_in_list)])  # (l_out + l_in) - |l_out - l_in| = 2*min(l_out, l_in)
            for j, (mul_in, l_in) in enumerate(zip(self.mul_in_list, self.l_in_list)):
                norm_coef[i, j] =  math.sqrt(4 * math.pi) * math.sqrt(2 * l_out + 1) / math.sqrt(num_summed_elements)

        self.norm_coef = norm_coef

    def forward(self, features, edge_index, radii, real_spherical_harmonics):
        radial_model_outputs = self.radial_model(radii)
        return self.propagate(edge_index, x=features, rsh=real_spherical_harmonics, rbf=radial_model_outputs)

    def message(self, x_j, rsh, rbf):
        return TensorMessageFunction.apply(x_j, rsh, rbf, self)


class TensorMessageFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, features, real_spherical_harmonics, radial_model_outputs, layer):
        F = features.transpose(0, 1).contiguous()
        Y = real_spherical_harmonics
        R = radial_model_outputs.transpose(0, 1).contiguous()

        msg = tensor_message.forward(
            layer.norm_coef,                                    # W
            layer.network.coupling_coefficients,                # C
            F,                                                  # F
            Y,                                                  # Y
            R,                                                  # R
            layer.l_out_list,                                   # L_out_list
            layer.l_in_list,                                    # L_in_list
            layer.mul_out_list,                                 # u_sizes
            layer.mul_in_list,                                  # v_sizes
            layer.grad_base_offsets,                            # output_base_offsets
            layer.network.coupling_coefficients_base_offsets,   # C_offsets
            layer.features_base_offsets,                        # F_base_offsets
            layer.R_base_offsets,                               # R_base_offsets
            layer.network.max_l_in + 1)                         # l_in_max_net_bound

        msg = msg.transpose_(0, 1).contiguous()                 # [(l_out, u, i), (a, b)] -> [(a, b), (l_out, u, i)]

        if features.requires_grad or radial_model_outputs.requires_grad:
            ctx.save_for_backward(F, Y, R)                      # F, R - transposed
            ctx.layer = layer

        return msg

    @staticmethod
    def backward(ctx, grad_output):
        F, Y, R = ctx.saved_tensors  # F, R - transposed
        layer = ctx.layer

        G = grad_output.transpose(0, 1).contiguous()

        msg_grad_F = tensor_message.backward_F(
            layer.norm_coef,                                    # W
            layer.network.coupling_coefficients,                # C
            G,                                                  # G
            Y,                                                  # Y
            R,                                                  # R
            layer.l_out_list,                                   # L_out_list
            layer.l_in_list,                                    # L_in_list
            layer.mul_out_list,                                 # u_sizes
            layer.mul_in_list,                                  # v_sizes
            layer.features_base_offsets,                        # output_base_offsets
            layer.network.coupling_coefficients_base_offsets,   # C_offsets
            layer.grad_base_offsets,                            # G_base_offsets
            layer.R_base_offsets,                               # R_base_offsets
            layer.network.max_l_in + 1)                         # l_in_max_net_bound

        msg_grad_F = msg_grad_F.transpose_(0, 1).contiguous()   # [(a, b), (l_in, v, j)]

        msg_grad_R = tensor_message.backward_R(
            layer.norm_coef,                                    # W
            layer.network.coupling_coefficients,                # C
            G,                                                  # G
            F,                                                  # F
            Y,                                                  # Y
            layer.l_out_list,                                   # L_out_list
            layer.l_in_list,                                    # L_in_list
            layer.mul_out_list,                                 # u_sizes
            layer.mul_in_list,                                  # v_sizes
            layer.R_base_offsets,                               # output_base_offsets
            layer.network.coupling_coefficients_base_offsets,   # C_offsets
            layer.grad_base_offsets,                            # G_base_offsets
            layer.features_base_offsets,                        # F_base_offsets
            layer.network.max_l_in + 1)                         # l_in_max_net_bound

        msg_grad_R = msg_grad_R.transpose_(0, 1).contiguous()   # [(a, b), (l_out, l_in, l, u, v)]

        return msg_grad_F, None, msg_grad_R, None
