import torch
from collections import defaultdict
import e3nn
import e3nn.SO3 as SO3


def split_Rs(Rs):
    """
    Given Rs with all same mulitplicity, split into tensor where multiplicity
    is separate channel.
    Returns:
    """
    muls, Ls = zip(*Rs)
    total = sum([m * (2 * L + 1) for m, L in Rs])
    single = sum([(2 * L + 1) for m, L in Rs])
    M = list(set(muls))[0]
    rearrange = torch.zeros(M, single, total)
    if len(set(muls)) != 1:
        raise ValueError("All L's must have same multiplicity.")
    # Assumes all L's have same multiplicity
    # Take [batch, N, C] to [batch, N, M, D]
    start = 0
    indices = []
    
    for L in Ls:
        rep = 2 * L + 1
        index = (start, start + rep)
        indices.append(index)
        start += rep * M
    
    for m in range(M):
        new = 0
        for L, index in zip(Ls, indices):
            rep = 2 * L + 1
            rearrange[m, new : new + rep, m * rep + index[0] : m * rep + index[1]] = torch.eye(rep)
            new += rep
            
    return rearrange, indices


def simplify_Rs(Rs):
    """
    Return simplifed Rs and transformation matrix.
    Currently ignores parity!
    """
    try:
        mults, Ls, ps = zip(*Rs)
    except:
        mults, Ls = zip(*Rs)
    totals = [mult * (2 * L + 1) for mult, L, p in Rs]
    shuffle = torch.zeros(sum(totals), sum(totals))

    # Get total number of multiplicities by L
    d = defaultdict(int)
    for mult, L, p in Rs:
        d[L] += mult

    # Rs_new grouped and sorted by L
    Rs_new = sorted([x[::-1] for x in d.items()], key=lambda x: x[1])
    new_mults, new_Ls = zip(*Rs_new)
    new_totals = [mult * (2 * L + 1) for mult, L in Rs_new]

    # indices for different mults
    tot_indices = [[sum(totals[0:i]), sum(totals[0:i + 1])] for i in range(len(totals))]
    new_tot_indices = [[sum(new_totals[0:i]), sum(new_totals[0:i + 1])] for i in range(len(new_totals))]

    # group indices by L
    d_t_i = defaultdict(list)
    for L, index in zip(Ls, tot_indices):
        d_t_i[L].append(index)

    #
    total_bounds = sorted(list(d_t_i.items()), key=lambda x: x[0])
    new_total_bounds = list(zip(new_Ls, new_tot_indices))

    for old_indices, (L, new_index) in zip(total_bounds, new_total_bounds):
        old_indices_list = [torch.arange(i[0], i[1]) for i in old_indices[1]]
        new_index_list = torch.arange(new_index[0], new_index[1])
        shuffle[new_index_list, torch.cat(old_indices_list)] = 1

    return Rs_new, shuffle


def get_truncated_shuffled_Q(Rs):
    L_max = max(L for mul, L in Rs)
    Rs_out, Q = SO3.reduce_tensor_product(Rs, Rs)

    Rs_new, shuffle = simplify_Rs(Rs_out)
    Rs_new_trunc = [(mul, L) for mul, L in Rs_new if L <= L_max]

    new_Q = torch.einsum('lk,ijk->ijl', (shuffle, Q))
    shape = new_Q.shape
    total = sum([mul * (2 * L + 1) for mul, L in Rs_new_trunc])

    new_Q_trunc = new_Q[:, :, :total]
    return Rs_new_trunc, new_Q_trunc
