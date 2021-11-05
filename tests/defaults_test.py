import e3nn


def test_opt_defaults():
    a = e3nn.o3.FullyConnectedTensorProduct("4x1o", "4x1o", "4x1o")
    b = e3nn.o3.Linear("4x1o", "4x1o")
    assert a._specialized_code
    assert not a._optimize_einsums
    assert not b._optimize_einsums
    old_defaults = e3nn.get_optimization_defaults()
    try:
        e3nn.set_optimization_defaults(optimize_einsums=True)
        a = e3nn.o3.FullyConnectedTensorProduct("4x1o", "4x1o", "4x1o")
        b = e3nn.o3.Linear("4x1o", "4x1o")
        assert a._specialized_code
        assert a._optimize_einsums
        assert b._optimize_einsums
    finally:
        e3nn.set_optimization_defaults(**old_defaults)
    a = e3nn.o3.FullyConnectedTensorProduct("4x1o", "4x1o", "4x1o")
    b = e3nn.o3.Linear("4x1o", "3x1o")
    assert a._specialized_code
    assert not a._optimize_einsums
    assert not b._optimize_einsums
