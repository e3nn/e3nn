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

    >>> import torch
    >>> x = CartesianTensor("ijk=-jik=-ikj")
    >>> x.from_cartesian(torch.ones(3, 3, 3))
    tensor([0.])
    """
    def __new__(cls, formula, p=-1):
        f = formula.split('=')[0].replace('-', '')
        def vector(g):
            q, k = g
            return (-1)**k * o3.quaternion_to_matrix(q)
        irreps, Q = o3.reduce_tensor(formula, **{i : vector for i in f})
        ret = super().__new__(cls, irreps)
        ret.formula = formula
        ret.num_index = len(f)
        ret.Q = Q
        return ret

    def from_cartesian(self, data):
        Q = self.Q.flatten(-self.num_index)
        return data.flatten(-self.num_index) @ Q.T
