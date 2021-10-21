from e3nn import o3


class CartesianTensor(o3.Irreps):
    r"""representation of a cartesian tensor into irreps

    Parameters
    ----------
    formula : str

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

    >>> x = CartesianTensor("ij=ji")
    >>> t = torch.arange(9).to(torch.float).view(3,3)
    >>> y = x.from_cartesian(t)
    >>> z = x.to_cartesian(y)
    >>> torch.allclose(z, (t + t.T)/2, atol=1e-5)
    True
    """
    # pylint: disable=abstract-method

    # These are set in __new__
    _rtp: o3.ReducedTensorProducts
    num_index: int

    def __new__(
        # pylint: disable=signature-differs
        cls,
        formula,
    ):
        f = formula.split("=")[0].replace("-", "")
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
        return self._rtp(*xs)  # pylint: disable=not-callable

    def to_cartesian(self, data):
        r"""convert irreps tensor to cartesian tensor

        This is the symmetry-aware inverse operation of ``from_cartesian()``.

        Parameters
        ----------
        data : `torch.Tensor`
            irreps tensor of shape ``(..., D)``, where D is the dimension of the irreps,
            i.e. ``D=self.dim``.

        Returns
        -------
        `torch.Tensor`
            cartesian tensor of shape ``(..., 3, 3, 3, ...)``
        """
        Q = self.change_of_basis()
        cartesian_tensor = data @ Q.flatten(-self.num_index)

        shape = list(data.shape[:-1]) + list(Q.shape[1:])
        cartesian_tensor = cartesian_tensor.view(shape)

        return cartesian_tensor

    def change_of_basis(self):
        r"""change of basis from cartesian tensor to irreps

        Returns
        -------
        `torch.Tensor`
            irreps tensor of shape ``(self.dim, 3, 3, 3, ...)``
        """
        return self._rtp.change_of_basis
