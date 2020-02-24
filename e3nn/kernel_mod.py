# pylint: disable=missing-docstring, line-too-long, invalid-name, arguments-differ, no-member
import math
import torch

import e3nn.SO3 as SO3


class Kernel(torch.nn.Module):
    def __init__(self, Rs_in, Rs_out, RadialModel, get_l_filters=SO3.selection_rule, sh=SO3.spherical_harmonics_xyz, normalization='norm'):
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

        self.Rs_in = SO3.simplifyRs(Rs_in)
        self.Rs_out = SO3.simplifyRs(Rs_out)

        def filters_with_parity(l_in, p_in, l_out, p_out):
            nonlocal get_l_filters
            return [l for l in get_l_filters(l_in, l_out) if p_out == 0 or p_in * (-1) ** l == p_out]

        self.get_l_filters = filters_with_parity
        self.sh = sh

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

        # Clebsch-Gordan for filter, input, output
        # Rs_filter contains all degrees of freedom and L's for filters
        # paths contains [l_in, l_out, l_filter for each filter channel]
        Rs_filter, filter_clebsch_gordan, paths = SO3.tensor_productRs(Rs_in,
                                                                       Rs_out,
                                                                       get_l_output=get_l_filters,
                                                                       paths=True)

        self.n_path = len(paths)

        # Helper matrix for spherical harmonics
        Rs_filter_sorted, sort_mix = SO3.sortRs(Rs_filter)
        Rs_filter_simplify = SO3.simplifyRs(Rs_filter_sorted)
        irrep_mix = SO3.map_irrep_to_Rs(Rs_filter_simplify)
        # Contract sort_mix with rep_mix
        ylm_mix = torch.einsum('ij,il->jl', sort_mix, irrep_mix) 
        

        # Create and sort mix matrix for radial functions
        rf_mix = SO3.map_mul_to_Rs(Rs_filter)

        ###########################################
        #TODO: Write normalization based on paths #
        ###########################################

        # Create the radial model: R+ -> R^n_path
        # It contains the learned parameters
        self.R = RadialModel(self.n_path)
        self.set_of_l_filters = [L for mul, L, p in Rs_filter_simplify]
        # Check l_filters and rep_mix have same dim
        assert sum([2 * L + 1 for L in self.set_of_l_filters]) == irrep_mix.shape[-1]

        # Register mapping matrix buffers
        self.register_buffer('filter_clebsch_gordan', filter_clebsch_gordan)
        self.register_buffer('ylm_mapping_matrix', ylm_mix)
        self.register_buffer('radial_mapping_matrix', rf_mix)
        #self.register_buffer('norm_coef', norm_coef)

    def __repr__(self):
        return "{name} ({Rs_in} -> {Rs_out})".format(
            name=self.__class__.__name__,
            Rs_in=SO3.formatRs(self.Rs_in),
            Rs_out=SO3.formatRs(self.Rs_out),
        )

    def forward(self, r):
        """
        :param r: tensor [..., 3]
        :return: tensor [..., l_out * mul_out * m_out, l_in * mul_in * m_in]
        """
        *size, xyz = r.size()
        assert xyz == 3
        r = r.reshape(-1, 3)

        # precompute all needed spherical harmonics
        Y = self.sh(self.set_of_l_filters, r)  # [irreps, batch]

        # use the radial model to fix all the degrees of freedom
        # note: for the normalization we assume that the variance of R[i] is one
        radii = r.norm(2, dim=1)  # [batch]
        R = self.R(radii)  # [batch, mul_dimRs(Rs_filter)] == [batch, self.n_paths]
        assert R.shape[-1] == self.n_path

        ylm_mix = self.ylm_mapping_matrix  # [dimRs(Rs_filter), irrep_dimRs(Rs_filter)]
        rf_mix = self.radial_mapping_matrix  # [dimRs(Rs_filter), mul_dimRs(Rs_filter)]
        cg = self.filter_clebsch_gordan  # [dimRs(Rs_filter), dimRs(Rs_in), dimRs(Rs_out)]

        R = torch.einsum('ij,zj->zi', rf_mix, R)
        Y = torch.einsum('ij,jz->zi', ylm_mix, Y)
        #TODO: Add norms
        kernel = torch.einsum('kij,zk,zk->zij', cg, R, Y)
        return kernel.view(*size, kernel.shape[1], kernel.shape[2])
