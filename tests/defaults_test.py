import e3nn


def test_opt_defaults():
    a = e3nn.o3.Linear("4x1o", "4x1o")
    assert a.tp._specialized_code
    assert a.tp._optimize_einsums
    old_defaults = e3nn.get_optimization_defaults()
    try:
        e3nn.set_optimization_defaults(optimize_einsums=False)
        a = e3nn.o3.Linear("4x1o", "4x1o")
        assert a.tp._specialized_code
        assert not a.tp._optimize_einsums
    finally:
        e3nn.set_optimization_defaults(**old_defaults)
    a = e3nn.o3.Linear("4x1o", "4x1o")
    assert a.tp._specialized_code
    assert a.tp._optimize_einsums
