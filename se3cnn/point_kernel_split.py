# pylint: disable=C, R, arguments-differ, no-member
import torch
import se3cnn.SO3 as SO3


class FiniteElementModel(torch.nn.Module):
    def __init__(self, position, basis, Model, out_dim):
        '''
        :param position: tensor [i, ...]
        :param basis: scalar function: tensor [a, ...] -> [a]
        :param Model: Class(d1, d2), trainable model: R^d1 -> R^d2
        :param out_dim: output dimension
        '''
        super().__init__()
        self.position = position
        self.basis = basis
        self.f = Model(len(position), out_dim)

    def forward(self, x):
        """
        :param x: tensor [batch, ...]
        :return: tensor [batch, dim]
        """
        diff = x.unsqueeze(1) - self.position.unsqueeze(0)  # [batch, i, ...]
        batch, n, *rest = diff.size()
        x = self.basis(diff.view(-1, *rest)).view(batch, n)  # [batch, i]
        return self.f(x)


class SE3PointKernel(torch.nn.Module):
    def __init__(self, Rs_in, Rs_out, RadialModel, J_filter_max=10, sh_backwardable=False):
        '''
        :param Rs_in: list of couple (multiplicity, representation order)
        :param Rs_out: list of couple (multiplicity, representation order)
        :param RadialModel: Class(d), trainable model: R -> R^d
        '''
        super().__init__()

        self.Rs_out = [(mul, l) for mul, l in Rs_out if mul >= 1]
        self.Rs_in = [(mul, l) for mul, l in Rs_in if mul >= 1]
        self.n_out = sum(mul * (2 * l + 1) for mul, l in self.Rs_out)
        self.n_in = sum(mul * (2 * l + 1) for mul, l in self.Rs_in)

        self.J_filter_max = J_filter_max
        self.sh_backwardable = sh_backwardable

        n = 0
        set_of_Js = set()

        for i, (mul_out, l_out) in enumerate(self.Rs_out):
            for j, (mul_in, l_in) in enumerate(self.Rs_in):
                Js = self.get_Js(l_in, l_out)

                # compute the number of degrees of freedom
                n += mul_out * mul_in * len(Js)

                # create the set of all spherical harmonics orders needed
                set_of_Js = set_of_Js.union(Js)

                # precompute the change of basis Q
                Q = [SO3.basis_transformation_Q_J(J, l_in, l_out) for J in Js]
                Q = torch.cat(Q, dim=1).view(2 * l_out + 1, 2 * l_in + 1, -1)  # [m_out, m_in, J * m]
                self.register_buffer("Q_{}_{}".format(i, j), Q.type(torch.get_default_dtype()))

        # create the radial model: R+ -> R^n
        # it contains the learned parameters
        self.f = RadialModel(n)

        self.set_of_Js = sorted(set_of_Js)

    def __repr__(self):
        return "{name} ({Rs_in} -> {Rs_out})".format(
            name=self.__class__.__name__,
            Rs_in=self.Rs_in,
            Rs_out=self.Rs_out,
        )

    def get_Js(self, l_in, l_out):
        Js = list(range(abs(l_in - l_out), l_in + l_out + 1))
        Js = [J for J in Js if J <= self.J_filter_max]
        return Js

    def forward(self, difference_matrix):
        """
        :param difference_matrix: tensor [[batch,] N, M, 3]
        :return: tensor [l_out * mul_out * m_out, l_in * mul_in * m_in, [batch,] N, M]
        """
        has_batch = difference_matrix.dim() == 4
        if not has_batch:
            difference_matrix = difference_matrix.unsqueeze(0)

        batch, N, M, _ = difference_matrix.size()

        kernel = difference_matrix.new_zeros(self.n_out, self.n_in, batch, N, M)

        # precompute all needed spherical harmonics
        sh = SO3.spherical_harmonics_xyz_backwardable if self.sh_backwardable else SO3.spherical_harmonics_xyz
        Ys = sh(self.set_of_Js, difference_matrix)  # [J * m, batch, N, M]

        # use the radial model to fix all the degrees of freedom
        radii = difference_matrix.norm(2, dim=-1).view(-1)  # [batch * N * M]
        weights = self.f(radii).view(batch, N, M, -1)  # [batch, N, M, l_out * l_in * mul_out * mul_in * J]
        begin_w = 0

        begin_out = 0
        for i, (mul_out, l_out) in enumerate(self.Rs_out):
            s_out = slice(begin_out, begin_out + mul_out * (2 * l_out + 1))

            begin_in = 0
            for j, (mul_in, l_in) in enumerate(self.Rs_in):
                s_in = slice(begin_in, begin_in + mul_in * (2 * l_in + 1))

                Js = self.get_Js(l_in, l_out)

                # extract the subset of the `weights` that corresponds to the couple (l_out, l_in)
                n = mul_out * mul_in * len(Js)
                w = weights[:, :, :, begin_w: begin_w + n].contiguous().view(batch, N, M, mul_out, mul_in, -1)  # [batch, N, M, mul_out, mul_in, J]
                begin_w += n

                Qs = getattr(self, "Q_{}_{}".format(i, j))  # [m_out, m_in, J * m]

                # note: I don't know if we can vectorize this for loop because [J * m] cannot be put into [J, m]
                K = 0
                for k, J in enumerate(Js):
                    tmp = sum(2 * l + 1 for l in self.set_of_Js if l < J)
                    Y = Ys[tmp: tmp + 2 * J + 1]  # [m, batch, N, M]

                    tmp = sum(2 * l + 1 for l in Js if l < J)
                    Q = Qs[:, :, tmp: tmp + 2 * J + 1]  # [m_out, m_in, m]

                    # note: The multiplication with `w` could also be done outside of the for loop
                    K += torch.einsum("ijr,rknm,knmuv->uivjknm", (Q, Y, w[..., k]))  # [mul_out, m_out, mul_in, m_in, batch, N, M]

                if K is not 0:
                    kernel[s_out, s_in] = K.contiguous().view_as(kernel[s_out, s_in])

                begin_in += mul_in * (2 * l_in + 1)
            begin_out += mul_out * (2 * l_out + 1)

        if not has_batch:
            kernel = kernel.squeeze(2)

        return kernel

