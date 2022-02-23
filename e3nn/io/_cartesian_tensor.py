from typing import Optional
from threading import local
import weakref
import copy

import torch

from e3nn import o3


# CartesianTensor shouldn't carry a torch.nn.Module child
# but we don't want to rebuild the RTP every call
# cache it here in thread local storage (to avoid
# possibility of weird threading bugs)
_RTP_CACHE = local()
_RTP_CACHE.cache = weakref.WeakKeyDictionary()  # Dict[_formula, Dict[(device, dtype), ReducedTensorProducts]]


# need something that gets garbage collected; raw str won't do
class _formula:
    formula: str

    def __init__(self, formula: str) -> None:
        super().__init__()
        self.formula = formula


def _make_rtp(formula: str, indices: str, device=torch.device('cpu'), dtype=torch.float64) -> o3.ReducedTensorProducts:
    base_key = (torch.device('cpu'), torch.float64)
    # build formula as a "singleton"
    formula_obj = None
    for f in _RTP_CACHE.cache.keys():
        if f.formula == formula:
            formula_obj = f
            break
    if formula_obj is None:
        # make a new one
        formula_obj = _formula(formula)
        # create a new base RTP
        rtp = o3.ReducedTensorProducts(formula, **{i: "1o" for i in indices}, dtype=torch.float64)
        # save it
        _RTP_CACHE.cache[formula_obj] = {base_key: rtp}
    # key for the inner dict
    device = torch.device(device)
    key = (device, dtype)
    inner_dict = _RTP_CACHE.cache[formula_obj]

    # get cached if we have it
    if key in inner_dict:
        return inner_dict[key]
    else:
        # get the base RTP
        rtp = inner_dict[base_key]
        # copy and move the build RTP to device/dtype
        rtp = copy.deepcopy(rtp)
        rtp = rtp.to(device=device, dtype=dtype)
        # cache it (it can't have been in the cache already if we made it past the first return)
        inner_dict[key] = rtp
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
        rtp = _make_rtp(formula=formula, indices=indices)
        ret = super().__new__(cls, rtp.irreps_out)
        ret.formula = formula
        ret.indices = indices
        return ret

    @staticmethod
    def reset_rtp_cache():
        """Empty the CartesianTensor ReducedTensorProduct cache"""
        _RTP_CACHE.cache.clear()

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
        rtp = self.reduced_tensor_products(data)
        Q = rtp.change_of_basis.flatten(-len(self.indices))
        return data.flatten(-len(self.indices)) @ Q.T

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
        rtp = self.reduced_tensor_products(xs[0])
        return rtp(*xs)  # pylint: disable=not-callable

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
