# pylint: disable=missing-docstring, line-too-long, invalid-name, arguments-differ, no-member
import math
import torch

import e3nn.o3 as o3
import e3nn.rs as rs


class Kernel(torch.nn.Module):
    def __init__(self, Rs_in, Rs_out, RadialModel,
                 get_l_filters=o3.selection_rule, sh=o3.spherical_harmonics_xyz, normalization='norm'):
        '''
        :param Rs_in: list of triplet (multiplicity, representation order, parity)
        :param Rs_out: list of triplet (multiplicity, representation order, parity)
        :param RadialModel: Class(d), trainable model: R -> R^d
        :param get_l_filters: function of signature (l_in, l_out) -> [l_filter]
        :param sh: spherical harmonics function of signature ([l_filter], xyz[..., 3]) -> Y[m, ...]
        :param normalization: either 'norm' or 'component'
        representation order = nonnegative integer
        parity = 0 (no parity), 1 (even), -1 (odd)
        '''
        super().__init__()

        self.Rs_in = rs.simplify(Rs_in)
        self.Rs_out = rs.simplify(Rs_out)

        def filters_with_parity(l_in, p_in, l_out, p_out):
            nonlocal get_l_filters
            return [l for l in get_l_filters(l_in, l_out) if p_out == 0 or p_in * (-1) ** l == p_out]

        self.get_l_filters = filters_with_parity
        self.sh = sh

        ## NORMALIZATION PRE-PROCESS
        assert isinstance(normalization, str), "normalization should be passed as a string value"
        assert normalization in ['norm', 'component'], "normalization needs to be 'norm' or 'component'"
        self.normalization = normalization

        def lm_normalization(l_out, l_in):
            # put 2l_in+1 to keep the norm of the m vector constant
            # put 2l_ou+1 to keep the variance of each m component constant
            # sum_m Y_m^2 = (2l+1)/(4pi)  and  norm(Q) = 1  implies that norm(QY) = sqrt(1/4pi)
            lm_norm = None
            if normalization == 'norm':
                lm_norm = math.sqrt(2 * l_in + 1) * math.sqrt(4 * math.pi)
            elif normalization == 'component':
                lm_norm = math.sqrt(2 * l_out + 1) * math.sqrt(4 * math.pi)
            return lm_norm

        ## TENSOR PRODUCT
        # Clebsch-Gordan for filter, input, output
        # Rs_filter contains all degrees of freedom and L's for filters
        # Because we give Rs_out and then Rs_in
        # paths contains [l_out, l_in, l_filter for each filter channel]
        Rs_filter, filter_clebsch_gordan, paths = rs.tensor_product(
            Rs_out, Rs_in, get_l_output=get_l_filters, paths=True)

        ## NORAMALIZATION
        num_summed_list = rs.num_summed_elements(paths)
        # Write normalization based on paths #
        norm_coef = torch.zeros((len(self.Rs_out), len(self.Rs_in), 2))
        for i, (mul_out, l_out, p_out) in enumerate(self.Rs_out):
            # consider that we sum a bunch of [lambda_(m_out)] vectors
            # we need to count how many of them we sum in order to normalize the network
            for j, (mul_in, l_in, p_in) in enumerate(self.Rs_in):
                # normalization assuming that each terms are of order 1 and uncorrelated
                norm_coef[i, j, 0] = lm_normalization(l_out, l_in) / math.sqrt(num_summed_list[i])
                norm_coef[i, j, 1] = lm_normalization(l_out, l_in) / math.sqrt(mul_in)
        full_norm_coef = torch.einsum('nmx,in,jm->ijx',
                                      norm_coef,
                                      rs.map_tuple_to_Rs(self.Rs_out),
                                      rs.map_tuple_to_Rs(self.Rs_in))

        ## HELPER MATRICES
        # Helper matrix for spherical harmonics
        Rs_filter_sorted, sort_mix = rs.sort(Rs_filter)
        Rs_filter_simplify = rs.simplify(Rs_filter_sorted)
        irrep_mix = rs.map_irrep_to_Rs(Rs_filter_simplify)
        # Contract sort_mix with rep_mix
        ylm_mix = torch.einsum('ij,il->jl', sort_mix, irrep_mix) 
        # Create and sort mix matrix for radial functions
        rf_mix = rs.map_mul_to_Rs(Rs_filter)
        
        ## RADIAL MODEL
        # Create the radial model: R+ -> R^n_path
        # It contains the learned parameters
        self.n_path = len(paths)
        self.R = RadialModel(self.n_path)
        self.set_of_l_filters = [L for mul, L, p in Rs_filter_simplify]
        # Check l_filters and rep_mix have same dim
        assert sum([2 * L + 1 for L in self.set_of_l_filters]) == irrep_mix.shape[-1]

        ## REGISTER BUFFERS
        # Register mapping matrix buffers and normalization coefficients
        self.register_buffer('filter_clebsch_gordan', filter_clebsch_gordan)
        self.register_buffer('ylm_mapping_matrix', ylm_mix)
        self.register_buffer('radial_mapping_matrix', rf_mix)
        self.register_buffer('norm_coef', full_norm_coef)

    def __repr__(self):
        return "{name} ({Rs_in} -> {Rs_out})".format(
            name=self.__class__.__name__,
            Rs_in=rs.format(self.Rs_in),
            Rs_out=rs.format(self.Rs_out),
        )

    def forward(self, r):
        """
        :param r: tensor [..., 3]
        :return: tensor [..., l_out * mul_out * m_out, l_in * mul_in * m_in]
        """
        *size, xyz = r.size()
        assert xyz == 3
        r = r.reshape(-1, 3)

        ## SPHERICAL HARMONICS
        # precompute all needed spherical harmonics
        Y = self.sh(self.set_of_l_filters, r)  # [irreps, batch]

        ## RADIAL MODEL
        # use the radial model to fix all the degrees of freedom
        # note: for the normalization we assume that the variance of R[i] is one
        radii = r.norm(2, dim=1)  # [batch]
        R = self.R(radii)  # [batch, mul_dimRs(Rs_filter)] == [batch, self.n_paths]
        assert R.shape[-1] == self.n_path

        ## HELPER AND CG MATRICES
        ylm_mix = self.ylm_mapping_matrix  # [dimRs(Rs_filter), irrep_dimRs(Rs_filter)]
        rf_mix = self.radial_mapping_matrix  # [dimRs(Rs_filter), mul_dimRs(Rs_filter)]
        cg = self.filter_clebsch_gordan  # [dimRs(Rs_filter), dimRs(Rs_in), dimRs(Rs_out)]

        ## COMPUTE
        R = torch.einsum('ij,zj->zi', rf_mix, R)
        Y = torch.einsum('ij,jz->zi', ylm_mix, Y)
        norm_coef = self.norm_coef[:, :, (radii == 0).type(torch.long)]
        kernel = torch.einsum('kij,zk,zk,ijz->zij', cg, R, Y, norm_coef)
        return kernel.view(*size, kernel.shape[1], kernel.shape[2])
