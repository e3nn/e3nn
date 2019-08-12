import torch

from se3cnn.util.default_dtype import torch_default_dtype


def parity_operator(Rs, dtype=None):
    """Returns a sum([2*l+1 for _, l, _ in Rs]) length mask. Even parity (1) masks as 1, Odd parity (-1) masks as -1."""
    assert all([len(i) == 3 for i in Rs])

    group_element = []

    if dtype is None:
        dtype = torch.get_default_dtype()

    with torch_default_dtype(dtype):
        for m, l, p in Rs:
            if p == 1:
                parity = torch.tensor(1).repeat(2 * l + 1).repeat(m)
            elif p == -1:
                parity = torch.tensor(-1).repeat(2 * l + 1).repeat(m)
            else:
                raise ValueError('Parity must be defined as -1 or 1 for each p in Rs.')
            group_element.append(parity)
        return torch.cat(group_element)


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
