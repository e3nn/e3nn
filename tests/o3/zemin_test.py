import torch

from e3nn import o3


def wigner_blocks(rotation, lmax):
    alpha, beta, gamma = o3.matrix_to_angles(rotation)
    size = (lmax + 1) ** 2
    out = rotation.new_zeros(rotation.shape[:-2] + (size, size))
    offset = 0
    for degree in range(lmax + 1):
        width = 2 * degree + 1
        out[..., offset : offset + width, offset : offset + width] = o3.wigner_D(
            degree,
            alpha,
            beta,
            gamma,
        )
        offset += width
    return out


def cartesian_dim(irreps):
    return sum(mul * 3**ir.l for mul, ir in irreps)


def test_ictd_path_matrices_are_orthonormal_and_complete() -> None:
    _, decompositions, cartesian_to_spherical, _ = o3.ICTD(
        3,
        decomposition=True,
        dtype=torch.float64,
    )

    eye = torch.eye(27, dtype=torch.float64)
    assert torch.allclose(sum(decompositions), eye, atol=1e-10, rtol=1e-10)

    for path_matrix in cartesian_to_spherical:
        width = path_matrix.shape[-1]
        assert torch.allclose(
            path_matrix.T @ path_matrix,
            torch.eye(width, dtype=torch.float64),
            atol=1e-10,
            rtol=1e-10,
        )


def test_cartesian_3j_shape_and_normalization() -> None:
    z = o3.cartesian_3j(1, 1, 2, dtype=torch.float64)

    assert z.shape == (3, 3, 9)
    assert torch.allclose(
        torch.linalg.vector_norm(z),
        torch.tensor(1.0, dtype=torch.float64),
    )


def test_cartesian_harmonics_accepts_spherical_irreps_with_cartesian_dim() -> None:
    irreps = o3.Irreps.spherical_harmonics(3)
    x = torch.randn(2, 5, 3, dtype=torch.float64)

    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        harmonics = o3.CartesianHarmonics(irreps, normalize=True)
    finally:
        torch.set_default_dtype(previous)

    out = harmonics(x)

    assert harmonics.irreps_out == irreps
    assert out.shape == (2, 5, cartesian_dim(irreps))

    rank_2_start = 1 + 3
    rank_2 = out[..., rank_2_start : rank_2_start + 9].reshape(2, 5, 3, 3)
    assert torch.allclose(
        torch.einsum("...ii->...", rank_2),
        torch.zeros(2, 5, dtype=torch.float64),
        atol=1e-10,
    )


def test_cartesian_harmonics_slices_noncontiguous_spherical_irreps() -> None:
    irreps = o3.Irreps("1x0e + 1x2e")
    x = torch.randn(5, 3, dtype=torch.float64)

    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        harmonics = o3.CartesianHarmonics(irreps, normalize=True)
    finally:
        torch.set_default_dtype(previous)

    out = harmonics(x)

    assert harmonics.irreps_out == irreps
    assert out.shape == (5, cartesian_dim(irreps))
    assert torch.allclose(out[:, :1], torch.ones(5, 1, dtype=torch.float64))

    rank_2 = out[:, 1:].reshape(5, 3, 3)
    assert torch.allclose(
        torch.einsum("bii->b", rank_2),
        torch.zeros(5, dtype=torch.float64),
        atol=1e-10,
    )


def test_cartesian_wigner_d_methods_match_e3nn_wigner_d(float_tolerance) -> None:
    rotation = o3.rand_matrix(7, dtype=torch.float64)

    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        default = o3.CartesianWignerD(4)(rotation)
        recursive = o3.CartesianWignerD(4, wigner_type="recursive")(rotation)
        direct = o3.CartesianWignerD(4, wigner_type="direct")(rotation)
        expected = wigner_blocks(rotation, 4)
    finally:
        torch.set_default_dtype(previous)

    assert torch.allclose(default, recursive, atol=1e-10, rtol=1e-10)
    assert torch.allclose(recursive, direct, atol=1e-10, rtol=1e-10)

    tolerance = float(float_tolerance)
    assert torch.allclose(recursive, expected, atol=tolerance, rtol=tolerance)
    assert torch.allclose(direct, expected, atol=tolerance, rtol=tolerance)
