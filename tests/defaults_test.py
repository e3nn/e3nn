import e3nn


def test_opt_defaults():
    a = e3nn.o3.FullyConnectedTensorProduct("4x1o", "4x1o", "4x1o")
    b = e3nn.o3.Linear("4x1o", "4x1o")
    assert a.compile_options["specialized_code"]
    assert a.compile_options["optimize_einsums"]
    assert b.compile_options["optimize_einsums"]
    old_defaults = e3nn.get_optimization_defaults()
    try:
        e3nn.set_optimization_defaults(optimize_einsums=False)
        a = e3nn.o3.FullyConnectedTensorProduct("4x1o", "4x1o", "4x1o")
        b = e3nn.o3.Linear("4x1o", "4x1o")
        assert a.compile_options["specialized_code"]
        assert not a.compile_options["optimize_einsums"]
        assert not b.compile_options["optimize_einsums"]
    finally:
        e3nn.set_optimization_defaults(**old_defaults)
    a = e3nn.o3.FullyConnectedTensorProduct("4x1o", "4x1o", "4x1o")
    b = e3nn.o3.Linear("4x1o", "3x1o")
    assert a.compile_options["specialized_code"]
    assert a.compile_options["optimize_einsums"]
    assert b.compile_options["optimize_einsums"]
