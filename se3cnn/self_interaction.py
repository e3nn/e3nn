import torch


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
                                torch.ones(self.dims_out[i],
                                           self.dims_in[i]))).view(
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
        return output
