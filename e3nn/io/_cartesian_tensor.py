from typing import Optional
from threading import local
from collections import defaultdict

import torch

from e3nn import o3


# CartesianTensor shouldn't carry a torch.nn.Module child
# but we don't want to rebuild the RTP every call
# cache it here in thread local storage (to avoid
# possibility of weird threading bugs)
_RTP_CACHE = local()
_RTP_CACHE.cache = defaultdict(dict)  # Dict[formula, Dict[(device, dtype), ReducedTensorProducts]]
_RTP_CACHE.refcount = defaultdict(lambda: 0)  # Dict[formula, int]


def _make_rtp(formula: str, indices: str, device, dtype) -> o3.ReducedTensorProducts:
    device = torch.device(device)
    key = (device, dtype)
    # TODO: this can be done a little more intellegently by deepcopying
    # an existing RTP for the given formula, and moving it to device and dtype
    # For standard workflows, however, this should only happen twice anyway
    # and it shouldn't be a big deal.
    # This way we at least avoid constant CPU-GPU traffic if a formula is used
    # on both CPU and GPU.
    if key in _RTP_CACHE.cache.get(formula, {}):
        return _RTP_CACHE.cache[formula][key]
    else:
        rtp = o3.ReducedTensorProducts(formula, **{i: "1o" for i in indices})
        rtp = rtp.to(device=device, dtype=dtype)
        _RTP_CACHE.cache[formula][key] = rtp
        return rtp


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
    formula: str
    indices: str

    def __new__(
        # pylint: disable=signature-differs
        cls,
        formula,
    ):
        indices = formula.split("=")[0].replace("-", "")
        rtp = _make_rtp(formula=formula, indices=indices, device="cpu", dtype=torch.get_default_dtype())
        _RTP_CACHE.refcount[formula] += 1
        ret = super().__new__(cls, rtp.irreps_out)
        ret.formula = formula
        ret.indices = indices
        return ret

    def __del__(self):
        # be polite and clean up cached RTPs if no current tensors
        # with given formula
        _RTP_CACHE.refcount[self.formula] -= 1
        if _RTP_CACHE.refcount[self.formula] <= 0:
            _RTP_CACHE.refcount[self.formula] = 0
            # rather than del in case of some weird case meaning its not there
            _RTP_CACHE.cache.pop(self.formula, None)

    @staticmethod
    def reset_rtp_cache():
        """Empty the CartesianTensor ReducedTensorProduct cache"""
        _RTP_CACHE.refcount.clear()
        _RTP_CACHE.cache.clear()

    def from_cartesian(self, data, rtp=None):
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
        if rtp is None:
            rtp = self.reduced_tensor_products(data)

        Q = rtp.change_of_basis.flatten(-len(self.indices))
        return data.flatten(-len(self.indices)) @ Q.T

    def from_vectors(self, *xs, rtp=None):
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
        if rtp is None:
            rtp = self.reduced_tensor_products(xs[0])

        return rtp(*xs)  # pylint: disable=not-callable

    def to_cartesian(self, data, rtp=None):
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
        if rtp is None:
            rtp = self.reduced_tensor_products(data)

        Q = rtp.change_of_basis
        cartesian_tensor = data @ Q.flatten(-len(self.indices))

        shape = list(data.shape[:-1]) + list(Q.shape[1:])
        cartesian_tensor = cartesian_tensor.view(shape)

        return cartesian_tensor

    def reduced_tensor_products(self, data: Optional[torch.Tensor] = None, device=None, dtype=None) -> o3.ReducedTensorProducts:
        r"""Get the reduced tensor products for this ``CartesianTensor``.

        Looks for a caches RTP and creates it if one does not exist.

        Parameters
        ----------
        data : `torch.Tensor`
            an example tensor from which to take the device and dtype for the RTP
        device : `torch.device`
            the device for the RTP
        dtype : `torch.dtype`
            the dtype for the RTP

        Returns
        -------
        `e3nn.ReducedTensorProducts`
            reduced tensor products
        """
        if data is not None:
            assert device is None and dtype is None
            device = data.device
            dtype = data.dtype
        else:
            device = "cpu"
            dtype = torch.get_default_dtype()
        return _make_rtp(formula=self.formula, indices=self.indices, device=device, dtype=dtype)
