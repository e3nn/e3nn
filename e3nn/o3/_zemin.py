r"""irreducible Cartesian tensor utilities."""

from typing import List, Union, Any

import torch

from e3nn.util import explicit_default_types
from ._wigner import wigner_3j
from ._irreps import Irreps


def factorial(n: int) -> int:
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result


def double_factorial(n: int) -> int:
    result = 1
    for i in range(n, 0, -2):
        result *= i
    return result


def _norm(n: int) -> float:
    return double_factorial(2 * n - 1) / factorial(n)


def expand_dims_to(T: torch.Tensor, n_dim: int, dim: int = -1) -> torch.Tensor:
    while T.ndim < n_dim:
        T = T.unsqueeze(dim)
    return T


def ICTD(
    n_total: int,
    w: int = -1,
    decomposition: bool = True,
    dtype = None,
    device = None,
):
    r"""Irreducible Cartesian tensor decomposition.

    See https://arxiv.org/abs/2412.18263 for more details.
    The ICTD construction follows the parentage scheme described in the ICTD
    paper: path matrices are built by chain-like contractions with
    Clebsch-Gordan coefficients. Their columns form orthonormal changes of basis
    between Cartesian tensor products and spherical direct-sum components, so
    an ICT projection matrix is ``C @ C.T`` for a path matrix ``C``.

    Parameters
    ----------
    n_total : int
        Rank of the Cartesian tensor product space ``(R^3)^{\otimes n_total}``.
    w : int, optional
        If ``-1``, return all paths.  Otherwise return the first parentage path
        whose final weight is ``w``.
    decomposition : bool, optional
        If ``True``, also return ICT projection matrices ``C @ C.T``.
    dtype, device : optional
        Output dtype and device.

    Returns
    -------
    paths, decompositions, cartesian_to_spherical, spherical_to_cartesian
        ``cartesian_to_spherical`` contains path matrices ``C`` of shape
        ``(3**n_total, 2 * l + 1)``.  ``spherical_to_cartesian`` contains
        ``C.T``.  When ``decomposition=False``, ``decompositions`` is empty.
    """

    assert isinstance(n_total, int) and n_total >= 0
    assert w == -1 or isinstance(w, int) and 0 <= w <= n_total

    dtype, device = explicit_default_types(dtype, device)

    n_now = 0
    j_now = 0
    path_list = []
    this_path = []
    this_pathmatrix = wigner_3j(0, 0, 0, dtype=dtype, device=device)
    pathmatrices_list = []
    cart2sph_list = []
    sph2cart_list = []
    stop_flag = {"stop": False}

    # generate paths and path matrices
    def paths_generate(n_now, j_now, this_path, this_pathmatrix, n_total):
        if stop_flag["stop"]:
            return
        if n_now <= n_total:
            if stop_flag["stop"]:
                return
            this_path.append(j_now)
            for j in [j_now + 1, j_now, j_now - 1]:
                if not (j_now == 0 and (j != 1)) and n_now + 1 <= n_total:
                    cgmatrix = wigner_3j(1, j_now, j, dtype=dtype, device=device)
                    this_pathmatrix_ = torch.einsum(
                        "abc,dce->dabe", this_pathmatrix, cgmatrix
                    )
                    this_pathmatrix_ = this_pathmatrix_.reshape(
                        cgmatrix.shape[0], -1, cgmatrix.shape[-1]
                    )
                    paths_generate(
                        n_now + 1, j, this_path.copy(), this_pathmatrix_, n_total
                    )
            if n_now == n_total:
                this_pathmatrix = this_pathmatrix.reshape(-1, this_pathmatrix.shape[-1])
                this_pathmatrix = this_pathmatrix * (
                    1.0 / (this_pathmatrix**2).sum(0)[0] ** (0.5)
                )  # normalize
                if w == -1:
                    pathmatrices_list.append(this_pathmatrix)
                    path_list.append(this_path)
                else:
                    if path_list:
                        path_list[-1] = this_path
                        pathmatrices_list[-1] = this_pathmatrix
                    else:
                        pathmatrices_list.append(this_pathmatrix)
                        path_list.append(this_path)
                    if this_path[-1] == w:
                        stop_flag["stop"] = True
                        return
        return
    paths_generate(n_now, j_now, this_path, this_pathmatrix, n_total)
    decomp_list = []
    for path_matrix in pathmatrices_list:
        if decomposition:
            decomp_list.append(path_matrix @ path_matrix.T)
        cart2sph_list.append(path_matrix)
        sph2cart_list.append(path_matrix.T)
    return path_list, decomp_list, cart2sph_list, sph2cart_list


def cartesian_3j(l1: int, l2: int, l3: int, dtype = None, device = None) -> torch.Tensor:
    r"""Cartesian analogue of the Wigner-3j.

    See https://arxiv.org/abs/2512.16882 for more details.
    Cartesian-3j is obtained by contracting ``wigner_3j(l1, l2, l3)`` 
    with the three highest-weight ICTD path matrices.
    More generally, contracting over part of the path matrices yields direct 
    couplings between Cartesian and spherical tensors.

    Returns
    -------
    `torch.Tensor`
        A normalized tensor of shape ``(3**l1, 3**l2, 3**l3)``.
    """

    assert abs(l2 - l3) <= l1 <= l2 + l3
    assert isinstance(l1, int) and isinstance(l2, int) and isinstance(l3, int)

    dtype, device = explicit_default_types(dtype, device)
    C = torch.einsum(
        "ijk,ai,bj,ck->abc",
        wigner_3j(l1, l2, l3, dtype=dtype, device=device),
        ICTD(l1, l1, decomposition=False, dtype=dtype, device=device)[2][0],
        ICTD(l2, l2, decomposition=False, dtype=dtype, device=device)[2][0],
        ICTD(l3, l3, decomposition=False, dtype=dtype, device=device)[2][0],
    )
    C = C / torch.norm(C)
    return C.to(dtype=dtype, device=device, copy=True, memory_format=torch.contiguous_format)


class CartesianHarmonics(torch.nn.Module):
    r"""Cartesian harmonics built from highest-weight ICT projections.

    See https://arxiv.org/abs/2509.14961 for more details.
    CartesianHarmonics are defines from a unit vector
    ``x / ||x||`` by forming ``x^{\otimes l}``, multiplying by
    ``(2l - 1)!! / l!``, and applying the highest-weight ICTD projector. The
    result is a fully symmetric, traceless Cartesian tensor.
    """

    norm: bool
    traceless: bool 
    normalize: bool
    _ls_list: List[int]
    _lmax: int
    _is_range_lmax: bool
    _prof_str: str
    _slice_start_list: List[int]
    _slice_stop_list: List[int]
    eps: float

    def __init__(
        self,
        irreps_out: Union[int, List[int], str, Irreps],
        normalize: bool,
        irreps_in: Any = None,
        norm: bool = True,
        traceless: bool = True,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()

        if isinstance(irreps_out, str):
            irreps_out = Irreps(irreps_out)
        if isinstance(irreps_out, Irreps) and irreps_in is None:
            for mul, (l, p) in irreps_out:
                if l % 2 == 1 and p == 1:
                    irreps_in = Irreps("1e")
        if irreps_in is None:
            irreps_in = Irreps("1o")

        irreps_in = Irreps(irreps_in)
        if irreps_in not in (Irreps("1x1o"), Irreps("1x1e")):
            raise ValueError(
                f"irreps_in for SphericalHarmonics must be either a vector (`1x1o`) or a pseudovector (`1x1e`), "
                f"not `{irreps_in}`"
            )
        self.irreps_in = irreps_in
        input_p = irreps_in[0].ir.p

        if isinstance(irreps_out, Irreps):
            ls = []
            for mul, (l, p) in irreps_out:
                if p != input_p**l:
                    raise ValueError(
                        f"irreps_out `{irreps_out}` passed to SphericalHarmonics asked for an output of l = {l} with parity "
                        f"p = {p}, which is inconsistent with the input parity {input_p} — the output parity should have been "
                        f"p = {input_p**l}"
                    )
                ls.extend([l] * mul)
        elif isinstance(irreps_out, int):
            ls = [irreps_out]
        else:
            ls = list(irreps_out)

        _slice_start_list = []
        _slice_stop_list = []
        for l in ls:
            start = sum(3**i for i in range(l))
            stop = start + 3**l
            _slice_start_list.append(start)
            _slice_stop_list.append(stop)

        irreps_out = Irreps([(1, (l, input_p**l)) for l in ls]).simplify()
        self.irreps_out = irreps_out
        self._ls_list = ls
        self._lmax = max(ls)
        self._is_range_lmax = ls == list(range(max(ls) + 1))
        self._prof_str = f"cartesian_harmonics({ls})"
        self.normalize = normalize
        self.norm = norm
        self.traceless = traceless
        self._slice_start_list = _slice_start_list
        self._slice_stop_list = _slice_stop_list
        self.eps = eps
    
        for l in range(self._lmax+1):
            _, _, C, CT = ICTD(l, l, decomposition=False)
            self.register_buffer(f"C_{l}", C[0].to(torch.get_default_dtype()), persistent=False)
            self.register_buffer(f"CT_{l}", CT[0].to(torch.get_default_dtype()), persistent=False)

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        if self.normalize:
            v = torch.nn.functional.normalize(v, dim=-1, eps=self.eps)
        T = torch.ones_like(v[..., 0])
        leading_shape = v.shape[:-1]
        edge_attrs: List[torch.Tensor] = []
        edge_attrs.append(T.reshape(*leading_shape, -1))

        for l in range(1, self._lmax+1):
            T = T[..., None] * expand_dims_to(v, T.ndim + 1, dim=v.ndim - 1)
            edge_attrs.append(T.reshape(*leading_shape, -1))

        for l in range(1, self._lmax+1):
            T = edge_attrs[l]
            if self.norm:
                T = T * _norm(l)
            if self.traceless:
                if T.numel() != 0:
                    C = self.C(l).to(dtype=v.dtype, device=v.device)
                    CT = self.CT(l).to(dtype=v.dtype, device=v.device)
                    T = T @ C @ CT
            edge_attrs[l] = T

        ch = torch.cat(edge_attrs, dim=-1)
        if not self._is_range_lmax:
            ch = torch.cat(
                [
                    ch[..., start:stop] 
                    for start, stop in zip(self._slice_start_list, self._slice_stop_list)
                ], 
                dim=-1
            )
        return ch
    
    def C(self, l: int):
        return dict(self.named_buffers())[f"C_{l}"]
    
    def CT(self, l: int):
        return dict(self.named_buffers())[f"CT_{l}"]
    
    def __repr__(self):
        return (
            f"{self.__class__.__name__}(irreps_out={self.irreps_out}, "
            f"normalize={self.normalize}, norm={self.norm}, "
            f"traceless={self.traceless})"
        )


class CartesianWignerD(torch.nn.Module):
    r"""Wigner-D blocks computed from the relationship between Cartesian and spherical tensors.
    
    See https://arxiv.org/abs/2607.10664 for more details.
    ``wigner_type="recursive"`` uses the Wigner-3j recurrence implemented.
    ``wigner_type="direct"`` rotates highest-weight Cartesian basis tensors and
    projects them back with path matrices. 
    """

    def __init__(self, lmax: int, wigner_type: str = "recursive"):
        super().__init__()

        self.lmax = lmax
        self.wigner_type = wigner_type
        if self.wigner_type not in {"recursive", "direct"}:
            raise ValueError(
                f"Unknown wigner_type={self.wigner_type!r}; "
                "expected 'recursive', 'direct'."
            )
        
        for l in range(2, self.lmax + 1):
            self.register_buffer(f"CG_{l}", wigner_3j(1, l - 1, l), persistent=False)
        for l in range(self.lmax + 1):
            _, _, C, _ = ICTD(l, l, False)
            self.register_buffer(f"C_{l}", C[0], persistent=False)

    def forward(self, rot_mat: torch.Tensor) -> torch.Tensor:
        wigner = self._rotation_to_wigner_matrix(rot_mat, 0, self.lmax)
        return wigner
    
    def _rotation_to_wigner_matrix(self, rot_mat, start_lmax, end_lmax):
        if self.wigner_type == "direct":
            return self._rotation_to_wigner_matrix_direct(
                rot_mat,
                start_lmax,
                end_lmax,
            )
        return self._rotation_to_wigner_matrix_recursive(
            rot_mat,
            start_lmax,
            end_lmax,
        )

    def _rotate_cartesian_tensor(
        self,
        cartesian_basis: torch.Tensor,
        rotation: torch.Tensor,
        degree: int,
    ) -> torch.Tensor:
        out = cartesian_basis
        for axis in range(1, degree + 1):
            out = out.movedim(axis, -1)
            out = torch.einsum("bij,b...j->b...i", rotation, out)
            out = out.movedim(-1, axis)
        return out

    def _rotation_to_wigner_matrix_direct(
        self,
        edge_rot_mat: torch.Tensor,
        start_lmax: int,
        end_lmax: int,
    ) -> torch.Tensor:
        blocks = []
        batch = edge_rot_mat.shape[0]
        for degree in range(start_lmax, end_lmax + 1):
            C = self.C(degree).to(dtype=edge_rot_mat.dtype, device=edge_rot_mat.device)
            width = 2 * degree + 1
            cartesian_basis = C.unsqueeze(0).expand(batch, -1, -1)
            cartesian_basis = cartesian_basis.reshape(batch, *([3] * degree), width)
            rotated_basis = self._rotate_cartesian_tensor(
                cartesian_basis,
                edge_rot_mat,
                degree,
            )
            rotated_basis = rotated_basis.reshape(batch, 3**degree, width)
            blocks.append(torch.einsum("pi,bpj->bij", C, rotated_basis))

        size = int((end_lmax + 1) ** 2) - int(start_lmax**2)
        wigner = edge_rot_mat.new_zeros(batch, size, size)
        offset = 0
        for block in blocks:
            width = block.shape[-1]
            wigner[:, offset : offset + width, offset : offset + width] = block
            offset += width
        return wigner

    def _rotation_to_wigner_matrix_recursive(
        self,
        rot_mat: torch.Tensor,
        start_lmax: int,
        end_lmax: int,
    ) -> torch.Tensor:
        batch = rot_mat.shape[0]
        all_blocks = [rot_mat.new_ones(batch, 1, 1)]
        if end_lmax >= 1:
            all_blocks.append(rot_mat)
        for degree in range(2, end_lmax + 1):
            CG = self.CG(degree).to(dtype=rot_mat.dtype, device=rot_mat.device)
            left = torch.einsum("abm,eac->ebmc", CG, all_blocks[1])
            left = torch.einsum("ebmc,ebd->emcd", left, all_blocks[degree - 1])
            block = torch.einsum("emcd,cdn->emn", left, CG)
            all_blocks.append(block * (2 * degree + 1))
        blocks = all_blocks[start_lmax : end_lmax + 1]
        size = int((end_lmax + 1) ** 2) - int(start_lmax**2)
        wigner = rot_mat.new_zeros(batch, size, size)
        offset = 0
        for block in blocks:
            width = block.shape[-1]
            wigner[:, offset : offset + width, offset : offset + width] = block
            offset += width
        return wigner
    
    def C(self, degree: int) -> torch.Tensor:
        return dict(self.named_buffers())[f"C_{degree}"]

    def CG(self, degree: int) -> torch.Tensor:
        return dict(self.named_buffers())[f"CG_{degree}"]

    def extra_repr(self):
        return 'lmax={}, wigner_type={}'.format(self.lmax, self.wigner_type)
