import random
import copy
import tempfile

import pytest
import torch
from e3nn.o3 import TensorProduct, FullyConnectedTensorProduct, Irreps
from e3nn.util.test import assert_equivariant, assert_auto_jitable, assert_normalized


def make_tp(
    l1, p1, l2, p2, lo, po, mode, weight,
    mul: int = 25,
    path_weights: bool = True,
    **kwargs
):
    def mul_out(mul):
        if mode == "uvuv":
            return mul**2
        if mode == "uvu<v":
            return mul * (mul - 1) // 2
        return mul

    try:
        return TensorProduct(
            [(mul, (l1, p1)), (19, (l1, p1))],
            [(mul, (l2, p2)), (19, (l2, p2))],
            [(mul_out(mul), (lo, po)), (mul_out(19), (lo, po))],
            [
                (0, 0, 0, mode, weight),
                (1, 1, 1, mode, weight),
                (0, 0, 1, 'uvw', True, 0.5 if path_weights else 1.0),
                (0, 1, 1, 'uvw', True, 0.2 if path_weights else 1.0),
            ],
            compile_left_right=True,
            compile_right=True,
            **kwargs
        )
    except AssertionError:
        return None


def random_params(n=25):
    params = set()
    while len(params) < n:
        l1 = random.randint(0, 2)
        p1 = random.choice([-1, 1])
        l2 = random.randint(0, 2)
        p2 = random.choice([-1, 1])
        lo = random.randint(0, 2)
        po = random.choice([-1, 1])
        mode = random.choice(['uvw', 'uvu', 'uvv', 'uuw', 'uuu', 'uvuv'])
        weight = random.choice([True, False])
        if make_tp(l1, p1, l2, p2, lo, po, mode, weight) is not None:
            params.add((l1, p1, l2, p2, lo, po, mode, weight))
    return params


@pytest.mark.parametrize('l1, p1, l2, p2, lo, po, mode, weight', random_params())
def test_bilinear_right_variance_equivariance(float_tolerance, l1, p1, l2, p2, lo, po, mode, weight):
    eps = float_tolerance
    n = 1_500
    tol = 3.0

    m = make_tp(l1, p1, l2, p2, lo, po, mode, weight)

    # bilinear
    x1 = torch.randn(2, m.irreps_in1.dim)
    x2 = torch.randn(2, m.irreps_in1.dim)
    y1 = torch.randn(2, m.irreps_in2.dim)
    y2 = torch.randn(2, m.irreps_in2.dim)

    z1 = m(x1 + 1.7 * x2, y1 - y2)
    z2 = m(x1, y1 - y2) + 1.7 * m(x2, y1 - y2)
    z3 = m(x1 + 1.7 * x2, y1) - m(x1 + 1.7 * x2, y2)
    assert (z1 - z2).abs().max() < eps
    assert (z1 - z3).abs().max() < eps

    # right
    z1 = m(x1, y1)
    z2 = torch.einsum('zi,zij->zj', x1, m.right(y1))
    assert (z1 - z2).abs().max() < eps

    # variance
    x1 = torch.randn(n, m.irreps_in1.dim)
    y1 = torch.randn(n, m.irreps_in2.dim)
    z1 = m(x1, y1).var(0)
    assert z1.mean().log10().abs() < torch.tensor(tol).log10()

    # equivariance
    assert_equivariant(m, irreps_in=[m.irreps_in1, m.irreps_in2], irreps_out=m.irreps_out)

    if weight:
        # linear in weights
        w1 = m.weight.clone().normal_()
        w2 = m.weight.clone().normal_()
        z1 = m(x1, y1, weight=w1) + 1.5 * m(x1, y1, weight=w2)
        z2 = m(x1, y1, weight=w1 + 1.5 * w2)
        assert (z1 - z2).abs().max() < eps


# This is a fairly expensive test, so we don't run too many configs
@pytest.mark.parametrize('path_normalization', ['element', 'path'])
@pytest.mark.parametrize('l1, p1, l2, p2, lo, po, mode, weight', random_params(n=8))
def test_normalized(l1, p1, l2, p2, lo, po, mode, weight, path_normalization):
    if torch.get_default_dtype() != torch.float32:
        pytest.skip(
            "No reason to run expensive normalization tests again at float64 expense."
        )
    # Explicit fixed path weights screw with the output normalization,
    # so don't use them
    m = make_tp(l1, p1, l2, p2, lo, po, mode, weight, mul=5, path_weights=False, path_normalization=path_normalization)
    # normalization
    # n_weight, n_input has to be decently high to ensure statistical convergence
    # especially for uvuv
    assert_normalized(
        m,
        n_weight=100,
        n_input=10_000,
        atol=0.5
    )


def test_empty():
    m = TensorProduct(
        "0x0e + 1o + 2e",
        "0e + 1o + 2e",
        "0x0e + 1o",
        [
            (0, 0, 0, "uvw", True),
            (1, 1, 0, "uvw", True),
        ],
        compile_right=True,
    )
    x1, x2 = m.irreps_in1.randn(4, -1), m.irreps_in2.randn(4, -1)
    out = m(x1, x2)
    assert out.shape == (4, m.irreps_out.dim)
    assert torch.all(out == 0.0)  # no instruction leads to the 1o output
    m.right(x2)


@pytest.mark.parametrize('normalization', ['component', 'norm'])
@pytest.mark.parametrize(
    'mode,weighted',
    [
        ('uvw', True),
        ('uvu', True),
        ('uvu', False),
        ('uvv', True),
        ('uvv', False),
        ('uuu', True),
        ('uuu', False),
        ('uuw', True),
        ('uuw', False)
    ]
)
def test_specialized_code(normalization, mode, weighted, float_tolerance):
    irreps_in1 = Irreps('4x0e + 4x1e + 4x2e')
    irreps_in2 = Irreps('5x0e + 5x1e + 5x2e')
    irreps_out = Irreps('6x0e + 6x1e + 6x2e')

    if mode == 'uvu':
        irreps_out = irreps_in1
    elif mode == 'uvv':
        irreps_out = irreps_in2
    elif mode == 'uuu':
        irreps_in2 = irreps_in1
        irreps_out = irreps_in1
    elif mode == 'uuw':
        irreps_in2 = irreps_in1
        # When unweighted, uuw is a plain sum over u and requires an output mul of 1
        if not weighted:
            irreps_out = Irreps([(1, ir) for _, ir in irreps_out])

    ins = [
        (0, 0, 0, mode, weighted, 1.0),

        (0, 1, 1, mode, weighted, 1.0),
        (1, 0, 1, mode, weighted, 1.0),
        (1, 1, 0, mode, weighted, 1.0),

        (1, 1, 1, mode, weighted, 1.0),

        (0, 2, 2, mode, weighted, 1.0),
        (2, 0, 2, mode, weighted, 1.0),
        (2, 2, 0, mode, weighted, 1.0),

        (2, 1, 1, mode, weighted, 1.0),
    ]
    tp1 = TensorProduct(
        irreps_in1, irreps_in2, irreps_out,
        ins, irrep_normalization=normalization,
        compile_right=True,
        _specialized_code=False,
    )
    tp2 = TensorProduct(
        irreps_in1, irreps_in2, irreps_out,
        ins, irrep_normalization=normalization,
        compile_right=True,
        _specialized_code=True,
    )
    with torch.no_grad():
        tp2.weight[:] = tp1.weight

    x = irreps_in1.randn(3, -1)
    y = irreps_in2.randn(3, -1)
    assert (tp1(x, y) - tp2(x, y)).abs().max() < float_tolerance
    assert (tp1.right(y) - tp2.right(y)).abs().max() < float_tolerance


def test_empty_irreps():
    tp = FullyConnectedTensorProduct('0e + 1e', Irreps([]), '0e + 1e')
    out = tp(torch.randn(1, 2, 4), torch.randn(2, 1, 0))
    assert out.shape == (2, 2, 4)


def test_single_out():
    tp1 = TensorProduct(
        "5x0e", "5x0e", "5x0e",
        [(0, 0, 0, "uvw", True, 1.0)]
    )
    tp2 = TensorProduct(
        "5x0e", "5x0e", "5x0e + 3x0o",
        [(0, 0, 0, "uvw", True, 1.0)]
    )
    with torch.no_grad():
        tp2.weight[:] = tp1.weight
    x1, x2 = torch.randn(3, 5), torch.randn(3, 5)
    out1 = tp1(x1, x2)
    out2 = tp2(x1, x2)
    assert out1.shape == (3, 5)
    assert out2.shape == (3, 8)
    assert torch.allclose(out1, out2[:, :5])
    assert torch.all(out2[:, 5:] == 0)


def test_empty_inputs():
    tp = FullyConnectedTensorProduct('0e + 1e', '0e + 1e', '0e + 1e', compile_right=True)
    out = tp(torch.randn(2, 1, 0, 1, 4), torch.randn(1, 2, 0, 3, 4))
    assert out.shape == (2, 2, 0, 3, 4)

    out = tp.right(torch.randn(1, 2, 0, 3, 4))
    assert out.shape == (1, 2, 0, 3, 4, 4)


@pytest.mark.parametrize('l1, p1, l2, p2, lo, po, mode, weight', random_params(n=2))
@pytest.mark.parametrize('special_code', [True, False])
@pytest.mark.parametrize('opt_ein', [True, False])
def test_jit(l1, p1, l2, p2, lo, po, mode, weight, special_code, opt_ein):
    """Test the JIT.

    This test is seperate from test_optimizations to ensure that just jitting a model has minimal error if any.
    """
    orig_tp = make_tp(
        l1, p1, l2, p2, lo, po, mode, weight,
        _specialized_code=special_code,
        _optimize_einsums=opt_ein
    )
    opt_tp = assert_auto_jitable(orig_tp)

    # Confirm equivariance of optimized model
    assert_equivariant(
        opt_tp,
        irreps_in=[orig_tp.irreps_in1, orig_tp.irreps_in2],
        irreps_out=orig_tp.irreps_out
    )

    # Confirm that it gives same results
    x1 = orig_tp.irreps_in1.randn(2, -1)
    x2 = orig_tp.irreps_in2.randn(2, -1)
    # TorchScript should casue very little if any numerical error
    assert torch.allclose(
        orig_tp(x1, x2),
        opt_tp(x1, x2),
    )
    assert torch.allclose(
        orig_tp.right(x2),
        opt_tp.right(x2),
    )


@pytest.mark.parametrize('l1, p1, l2, p2, lo, po, mode, weight', random_params(n=4))
@pytest.mark.parametrize('special_code', [True, False])
@pytest.mark.parametrize('opt_ein', [True, False])
@pytest.mark.parametrize('jit', [True, False])
def test_optimizations(l1, p1, l2, p2, lo, po, mode, weight, special_code, opt_ein, jit, float_tolerance):
    orig_tp = make_tp(
        l1, p1, l2, p2, lo, po, mode, weight,
        _specialized_code=False,
        _optimize_einsums=False
    )
    opt_tp = make_tp(
        l1, p1, l2, p2, lo, po, mode, weight,
        _specialized_code=special_code,
        _optimize_einsums=opt_ein
    )
    # We don't use state_dict here since that contains things like wigners that can differ between optimized and unoptimized TPs
    with torch.no_grad():
        opt_tp.weight[:] = orig_tp.weight
    assert opt_tp._specialized_code == special_code
    assert opt_tp._optimize_einsums == opt_ein

    if jit:
        opt_tp = assert_auto_jitable(opt_tp)

    # Confirm equivariance of optimized model
    assert_equivariant(
        opt_tp,
        irreps_in=[orig_tp.irreps_in1, orig_tp.irreps_in2],
        irreps_out=orig_tp.irreps_out
    )

    # Confirm that it gives same results
    x1 = orig_tp.irreps_in1.randn(2, -1)
    x2 = orig_tp.irreps_in2.randn(2, -1)
    assert torch.allclose(
        orig_tp(x1, x2),
        opt_tp(x1, x2),
        atol=float_tolerance  # numerical optimizations can cause meaningful numerical error by changing operations
    )
    assert torch.allclose(
        orig_tp.right(x2),
        opt_tp.right(x2),
        atol=float_tolerance
    )

    # We also test .to(), even if only with a dtype, to ensure that various optimizations still always store constants in correct ways
    other_dtype = next(
        d for d in [torch.float32, torch.float64]
        if d != torch.get_default_dtype()
    )
    x1, x2 = x1.to(other_dtype), x2.to(other_dtype)
    opt_tp = opt_tp.to(other_dtype)
    assert opt_tp(x1, x2).dtype == other_dtype


def test_input_weights_python():
    irreps_in1 = Irreps("1e + 2e + 3x3o")
    irreps_in2 = Irreps("1e + 2e + 3x3o")
    irreps_out = Irreps("1e + 2e + 3x3o")
    # - shared_weights = False -
    m = FullyConnectedTensorProduct(
        irreps_in1,
        irreps_in2,
        irreps_out,
        internal_weights=False,
        shared_weights=False
    )
    bdim = random.randint(1, 3)
    x1 = irreps_in1.randn(bdim, -1)
    x2 = irreps_in2.randn(bdim, -1)
    w = [torch.randn((bdim,) + ins.path_shape) for ins in m.instructions if ins.has_weight]
    m(x1, x2, w)
    # - shared_weights = True -
    m = FullyConnectedTensorProduct(
        irreps_in1,
        irreps_in2,
        irreps_out,
        internal_weights=False,
        shared_weights=True
    )
    bdim = random.randint(1, 3)
    x1 = irreps_in1.randn(bdim, -1)
    x2 = irreps_in2.randn(bdim, -1)
    w = [torch.randn(ins.path_shape) for ins in m.instructions if ins.has_weight]
    m(x1, x2, w)


def test_input_weights_jit():
    irreps_in1 = Irreps("1e + 2e + 3x3o")
    irreps_in2 = Irreps("1e + 2e + 3x3o")
    irreps_out = Irreps("1e + 2e + 3x3o")
    # - shared_weights = False -
    m = FullyConnectedTensorProduct(
        irreps_in1,
        irreps_in2,
        irreps_out,
        internal_weights=False,
        shared_weights=False,
        compile_right=True,
    )
    traced = assert_auto_jitable(m)
    x1 = irreps_in1.randn(2, -1)
    x2 = irreps_in2.randn(2, -1)
    w = torch.randn(2, m.weight_numel)
    with pytest.raises((RuntimeError, torch.jit.Error)):
        m(x1, x2)  # it should require weights
    with pytest.raises((RuntimeError, torch.jit.Error)):
        traced(x1, x2)  # it should also require weights
    with pytest.raises((RuntimeError, torch.jit.Error)):
        traced(x1, x2, w[0])  # it should reject insufficient weights
    # Does the trace give right results?
    assert torch.allclose(
        m(x1, x2, w),
        traced(x1, x2, w)
    )

    # Confirm that weird batch dimensions give the same results
    for f in (m, traced):
        x1 = irreps_in1.randn(2, 1, 4, -1)
        x2 = irreps_in2.randn(2, 3, 1, -1)
        w = torch.randn(3, 4, f.weight_numel)
        assert torch.allclose(
            f(x1, x2, w).reshape(24, -1),
            f(x1.expand(2, 3, 4, -1).reshape(24, -1), x2.expand(2, 3, 4, -1).reshape(24, -1), w[None].expand(2, 3, 4, -1).reshape(24, -1))
        )
        assert torch.allclose(
            f.right(x2, w).reshape(24, -1),
            f.right(x2.expand(2, 3, 4, -1).reshape(24, -1), w[None].expand(2, 3, 4, -1).reshape(24, -1)).reshape(24, -1)
        )

    # - shared_weights = True -
    m = FullyConnectedTensorProduct(
        irreps_in1,
        irreps_in2,
        irreps_out,
        internal_weights=False,
        shared_weights=True
    )
    traced = assert_auto_jitable(m)
    w = torch.randn(m.weight_numel)
    with pytest.raises((RuntimeError, torch.jit.Error)):
        m(x1, x2)  # it should require weights
    with pytest.raises((RuntimeError, torch.jit.Error)):
        traced(x1, x2)  # it should also require weights
    with pytest.raises((RuntimeError, torch.jit.Error)):
        traced(x1, x2, torch.randn(2, m.weight_numel))  # it should reject too many weights
    # Does the trace give right results?
    assert torch.allclose(
        m(x1, x2, w),
        traced(x1, x2, w)
    )


def test_weight_view_for_instruction():
    irreps_in1 = Irreps("1e + 2e + 3x3o")
    irreps_in2 = Irreps("1e + 2e + 3x3o")
    irreps_out = Irreps("1e + 2e + 3x3o")
    x1 = irreps_in1.randn(2, -1)
    x2 = irreps_in2.randn(2, -1)
    m = FullyConnectedTensorProduct(irreps_in1, irreps_in2, irreps_out)

    # Find all paths to the first output
    ins_idexes = [i for i, ins in enumerate(m.instructions) if ins.i_out == 0]
    with torch.no_grad():
        for i in ins_idexes:
            m.weight_view_for_instruction(i).zero_()

    out = m(x1, x2)
    assert torch.all(out[:, :1] == 0.0)
    assert torch.any(out[:, 1:] > 0.0)


def test_weight_views():
    irreps_in1 = Irreps("1e + 2e + 3x3o")
    irreps_in2 = Irreps("1e + 2e + 3x3o")
    irreps_out = Irreps("1e + 2e + 3x3o")
    batchdim = 3
    x1 = irreps_in1.randn(batchdim, -1)
    x2 = irreps_in2.randn(batchdim, -1)
    # shared weights
    m = FullyConnectedTensorProduct(irreps_in1, irreps_in2, irreps_out)
    with torch.no_grad():
        for w in m.weight_views():
            w.zero_()
    assert torch.all(m(x1, x2) == 0.0)

    # unshared weights
    m = FullyConnectedTensorProduct(irreps_in1, irreps_in2, irreps_out, shared_weights=False)
    weights = torch.randn(batchdim, m.weight_numel)
    with torch.no_grad():
        for w in m.weight_views(weights):
            w.zero_()
    assert torch.all(m(x1, x2, weights) == 0.0)


@pytest.mark.parametrize('l1, p1, l2, p2, lo, po, mode, weight', random_params(n=1))
def test_deepcopy(l1, p1, l2, p2, lo, po, mode, weight):
    tp = make_tp(l1, p1, l2, p2, lo, po, mode, weight)
    x1 = torch.randn(2, tp.irreps_in1.dim)
    x2 = torch.randn(2, tp.irreps_in2.dim)
    res1 = tp(x1, x2)
    tp_copy = copy.deepcopy(tp)
    res2 = tp_copy(x1, x2)
    assert torch.allclose(res1, res2)


@pytest.mark.parametrize('l1, p1, l2, p2, lo, po, mode, weight', random_params(n=1))
def test_save(l1, p1, l2, p2, lo, po, mode, weight):
    tp = make_tp(l1, p1, l2, p2, lo, po, mode, weight)
    # Saved TP
    with tempfile.NamedTemporaryFile(suffix=".pth") as tmp:
        torch.save(tp, tmp.name)
        tp2 = torch.load(tmp.name)
    # JITed, saved TP
    with tempfile.NamedTemporaryFile(suffix=".pth") as tmp:
        tp_jit = assert_auto_jitable(tp)
        tp_jit.save(tmp.name)
        tp3 = torch.jit.load(tmp.name)
    # Double-saved TP
    with tempfile.NamedTemporaryFile(suffix=".pth") as tmp:
        torch.save(tp2, tmp.name)
        tp4 = torch.load(tmp.name)
    x1 = torch.randn(2, tp.irreps_in1.dim)
    x2 = torch.randn(2, tp.irreps_in2.dim)
    res1 = tp(x1, x2)
    res2 = tp2(x1, x2)
    res3 = tp3(x1, x2)
    res4 = tp4(x1, x2)
    assert torch.allclose(res1, res2)
    assert torch.allclose(res1, res3)
    assert torch.allclose(res1, res4)


def test_triu_mode():
    tp = TensorProduct("10x0e", "10x0e", "45x0e", [(0, 0, 0, "uvu<v", False)])
    tp(torch.randn(2, 10), torch.randn(2, 10))

    m = assert_auto_jitable(tp)

    assert_equivariant(m, irreps_in=[m.irreps_in1, m.irreps_in2], irreps_out=m.irreps_out)
