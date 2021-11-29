r"""Allows for clean lookup of Irreducible representations of :math:`O(3)`

Examples
--------
Create a scalar representation (:math:`l=0`) of even parity.

>>> from e3nn.o3 import irreps
>>> assert irreps.l1y == Irrep("1y")
True

>>> from e3nn.o3.irreps import l1o, l2o
>>> assert l1o + l2o == Irrep("1o") + Irrep("2o")
True
"""

from .._irreps import Irrep


def __getattr__(l: str):
    r"""Creates an Irreps obeject by reflection

        Parameters
        ----------
        l : string
            the o3 object name prefixed by l. Example: l1o == Irrep("1o")

        Returns
        -------
        `e3nn.o3.Irreps`
            representation of :math:`(Y^0, Y^1, \dots, Y^{\mathrm{lmax}})`
    """
    prefix, *name = l
    if prefix != "l" or not name:
        raise AssertionError("Attribute should match 'l.+'")
    return Irrep("".join(name))
