import torch
from collections import defaultdict

class ConcatenateSphericalSignals(torch.nn.Module):
    def __init__(self, Rs_in1, Rs_in2):
        super(ConcatenateSphericalSignals, self).__init__()
        features1 = sum([m * (2 * l + 1) for m, l in Rs_in1])
        features2 = sum([m * (2 * l + 1) for m, l in Rs_in2])
        
        self.Rs_in1 = Rs_in1
        self.Rs_in2 = Rs_in2
        
        
        features_out = defaultdict(int)
        for m, l in Rs_in1:
            features_out[l] += m
        for m, l in Rs_in2:
            features_out[l] += m
            
        L, M = zip(*features_out.items())
        self.Rs_out = list(zip(M, L))
        
        
        self.register_buffer('mixing_matrix', 
                             torch.zeros(features1 + features2, 
                                         features1 + features2))
        
        index1, index2 = 0, 0
        dim_index1, dim_index2, dim_index3 = 0, 0, 0
        
        while dim_index1 < features1 or dim_index2 < features2:
            if index1 < len(Rs_in1):
                m1, l1 = Rs_in1[index1]
            if index2 < len(Rs_in2):
                m2, l2 = Rs_in2[index2]
            if dim_index1 == features1:  # add features from signal2
                increment = m2 * (2 * l2 + 1)
                slice1 = slice(dim_index3, dim_index3 + increment)
                slice2 = slice(features1 + dim_index2,
                               features1 + dim_index2 + increment)
                self.mixing_matrix[slice1, slice2] = torch.eye(increment)
                dim_index3 += increment
                dim_index2 += increment
                index2 += 1
                
            elif dim_index2 == features2:  # add features from signal1
                increment = m1 * (2 * l1 + 1)
                slice1 = slice(dim_index3, 
                               dim_index3 + increment)
                slice2 = slice(dim_index1,
                               dim_index1 + increment)
                self.mixing_matrix[slice1, slice2] = torch.eye(increment)
                dim_index3 += increment
                dim_index1 += increment
                index1 += 1
                
            elif l1 > l2:  # add features from signal2
                increment = m2 * (2 * l2 + 1)
                slice1 = slice(dim_index3, 
                               dim_index3 + increment)
                slice2 = slice(features1 + dim_index2,
                               features1 + dim_index2 + increment)
                self.mixing_matrix[slice1, slice2] = torch.eye(increment)
                dim_index3 += increment
                dim_index2 += increment
                index2 += 1
            
            else:  # add features from signal1
                increment = m1 * (2 * l1 + 1)
                slice1 = slice(dim_index3, 
                               dim_index3 + increment)
                slice2 = slice(dim_index1,
                               dim_index1 + increment)
                self.mixing_matrix[slice1, slice2] = torch.eye(increment)
                dim_index3 += increment
                dim_index1 += increment
                index1 += 1
                
    def forward(self, signal1, signal2):
        combined = torch.cat((signal1, signal2), dim=-3)
        return torch.einsum('dc,ncba->ndba', (self.mixing_matrix, combined))


class SelfInteraction(torch.nn.Module):
    """
    SelfInteraction is a fully connected layer within a given L.

    It's basically the same as a 1x1x1 convolution.
    """
    def __init__(self, Rs_in, Rs_out):
        super().__init__()
        self.Rs_in = Rs_in
        self.Rs_out = Rs_out
        self.Ls = list(zip(*Rs_in))
        assert list(zip(*Rs_in))[1] == list(zip(*Rs_out))[1]  # L's must match
        self.multiplicities_out = [m for m, _ in self.Rs_out]
        self.multiplicities_in = [m for m, _ in self.Rs_in]
        self.dims_out = [2 * l + 1 for _, l in self.Rs_out]
        self.dims_in = [2 * l + 1 for _, l in self.Rs_in]
        self.n_out = sum([self.multiplicities_out[i] * self.dims_out[i] for i
                          in range(len(self.multiplicities_out))])
        self.n_in = sum([self.multiplicities_in[j] * self.dims_in[j] for j in
                         range(len(self.multiplicities_in))])
        self.nweights = sum([m_in * m_out for m_in, m_out in
                             zip(self.multiplicities_in,
                                 self.multiplicities_out)])
        self.weight = torch.nn.Parameter(torch.randn(self.nweights))

    def combination(self, weight):
        # Create kernel
        kernel = torch.zeros(self.n_out, self.n_in, dtype=self.weight.dtype)
        begin_i = 0
        begin_j = 0
        weight_index = 0
        for i, ((m_in, l_in), (m_out, l_out)) in enumerate(zip(self.Rs_in,
                                                               self.Rs_out)):
            si = slice(begin_i, begin_i + m_out * self.dims_out[i])
            sj = slice(begin_j, begin_j + m_in * self.dims_in[i])
            w = weight[weight_index: weight_index + m_out * m_in].view(m_out,
                                                                       m_in)
            # We use einsum to do transpose and tensordots at the same time
            kernel[si, sj] = torch.einsum(
                'dc,ij->dicj', (w,
                                torch.eye(self.dims_out[i]))).view(
                                    m_out * self.dims_out[i],
                                    m_in * self.dims_in[i])
            begin_j += m_in * self.dims_in[i]
            begin_i += m_out * self.dims_out[i]
            weight_index += m_out * m_in
        return kernel

    def forward(self, input):
        kernel = self.combination(self.weight)
        # No mask needed if assumed input values are zero.
        if len(input.size()) == 2:
            # No batch dimension
            output = torch.einsum('ca,dc->da', (input, kernel))
        elif len(input.size()) == 3:
            # Batch dimension
            output = torch.einsum('nca,dc->nda', (input, kernel))
        elif len(input.size()) == 4:
            # Multiple atom indices
            output = torch.einsum('ncba,dc->ndba', (input, kernel))
        return output
