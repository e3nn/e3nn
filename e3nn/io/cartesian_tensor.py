import torch
from e3nn import o3


class CartesianTensor(o3.Irreps):
    r"""representation of a cartesian tensor into irreps

    Parameters
    ----------
    formula : str

    p : {+1, -1}

    Examples
    --------

    >>> CartesianTensor("ij=-ji")
    1x1e

    >>> x = CartesianTensor("ijk=-jik=-ikj")
    >>> x.from_tensor(torch.ones(3, 3, 3))
    tensor([0.])

    >>> x.from_vectors(torch.ones(3), torch.ones(3), torch.ones(3))
    tensor([0.])
    """
    def __new__(cls, formula):
        f = formula.split('=')[0].replace('-', '')
        rtp = o3.ReducedTensorProducts(formula, **{i : "1o" for i in f})
        ret = super().__new__(cls, rtp.irreps_out)
        ret.formula = formula
        ret.num_index = len(f)
        ret._rtp = rtp
        return ret

    def from_tensor(self, data):
        Q = self.Q.flatten(-self.num_index)
        return data.flatten(-self.num_index) @ Q.T

    def from_vectors(self, *xs):
        A = torch.tensor([
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0.0],
        ])
        xs = [x @ A.T for x in xs]
        return self._rtp(*xs)

    def change_of_basis(self):
        A = torch.tensor([
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0.0],
        ])
        i = 'abcdefghijkl'[:self.num_index]
        j = ',am,bn,co,dp,eq,fr,gs,ht,iu,jv,kw,lx'[:3 * self.num_index]
        k = 'mnopqrstuvwx'[:self.num_index]
        return torch.einsum(f"z{i}{j}->z{k}", self._rtp.change_of_basis, *[A] * self.num_index)
