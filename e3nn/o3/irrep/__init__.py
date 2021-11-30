r"""Allows for clean lookup of Irreducible representations of :math:`O(3)`

Examples
--------
Create a scalar representation (:math:`l=0`) of even parity.

>>> from e3nn.o3 import irrep
>>> irrep.l0e == Irrep("0e")
True

>>> from e3nn.o3.irrep import l1o, l2o
>>> l1o + l2o == Irrep("1o") + Irrep("2o")
True
"""

from .._irreps import Irrep


def __getattr__(name: str):
    r"""Creates an Irreps obeject by reflection

        Parameters
        ----------
        name : string
            the o3 object name prefixed by l. Example: l1o == Irrep("1o")

        Returns
        -------
        `e3nn.o3.Irrep`
            irreducible representation of :math:`O(3)`
    """

    prefix, *ir = name
    if prefix != "l" or not ir:
        raise AttributeError(f"'e3nn.o3.irrep' module has no attribute '{name}'")

    try:
        return Irrep("".join(ir))
    except (ValueError, AssertionError):
        raise AttributeError(f"'e3nn.o3.irrep' module has no attribute '{name}'")
