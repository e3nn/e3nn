import torch

from se3cnn.util.default_dtype import torch_default_dtype


def spherical_harmonic_parity_operator(Rs, dtype=None):
    """The parity group element for the coefficients of a spherical harmonic signal."""
    assert all([len(i) == 2 for i in Rs])

    group_element = []

    if dtype is None:
        dtype = torch.get_default_dtype()

    with torch_default_dtype(dtype):
        for m, l in Rs:
            if l % 2 == 0:
                parity = torch.tensor(1).repeat(2*l+1).repeat(m)
            else:
                parity = torch.tensor(-1).repeat(2*l+1).repeat(m)
            group_element.append(parity)
        return torch.cat(group_element)
