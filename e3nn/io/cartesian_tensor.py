from e3nn import o3


class CartesianTensor(o3.Irreps):
    r"""representation of a cartesian tensor into irreps

    Parameters
    ----------
    formula : str

    p : {+1, -1}

    Examples
    --------

    >>> import torch
    >>> CartesianTensor("ij=-ji")
    1x1e

    >>> x = CartesianTensor("ijk=-jik=-ikj")
    >>> x.from_cartesian(torch.ones(3, 3, 3))
    tensor([0.])

    >>> x.from_vectors(torch.ones(3), torch.ones(3), torch.ones(3))
    tensor([0.])
    """
    def __new__(cls, formula):
        f = formula.split('=')[0].replace('-', '')
        rtp = o3.ReducedTensorProducts(formula, **{i: "1o" for i in f})
        ret = super().__new__(cls, rtp.irreps_out)
        ret.formula = formula
        ret.num_index = len(f)
        ret._rtp = rtp
        return ret

    def from_cartesian(self, data):
        r"""convert cartesian tensor into irreps

        Parameters
        ----------
        data : `torch.Tensor`
            cartesian tensor of shape ``(..., 3, 3, 3, ...)``

        Returns
        -------
        `torch.Tensor`
            irreps tensor of shape ``(..., self.dim)``
        """
        Q = self.change_of_basis().flatten(-self.num_index)
        return data.flatten(-self.num_index) @ Q.T

    def from_vectors(self, *xs):
        r"""convert :math:`x_1 \otimes x_2 \otimes x_3 \otimes \dots`

        Parameters
        ----------
        xs : list of `torch.Tensor`
            list of vectors of shape ``(..., 3)``

        Returns
        -------
        `torch.Tensor`
            irreps tensor of shape ``(..., self.dim)``
        """
        return self._rtp(*xs)

    def change_of_basis(self):
        r"""change of basis from cartesian tensor to irreps

        Returns
        -------
        `torch.Tensor`
            irreps tensor of shape ``(self.dim, 3, 3, 3, ...)``
        """
        return self._rtp.change_of_basis
