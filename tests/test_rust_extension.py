import numpy as np
import pytest

from warp.solver.get_energy_tensor import (
    takeFiniteDifference1,
    takeFiniteDifference2,
    _ricciT_loops,
)


def _import_rust_funcs():
    try:
        import warp_core  # type: ignore
    except Exception:
        pytest.skip("warp_core extension not available")
    funcs = []
    for name in [
        "take_finite_difference1",
        "take_finite_difference2",
        "ricci_t_loops",
    ]:
        if not hasattr(warp_core, name):
            pytest.skip(f"warp_core missing {name}")
        funcs.append(getattr(warp_core, name))
    return funcs


def test_take_finite_difference1_rust_matches_python():
    rust_f1, _, _ = _import_rust_funcs()
    tensor = np.random.rand(3, 4, 5)
    delta = [0.1, 0.2, 0.3]
    axis = 1
    py_res = takeFiniteDifference1(tensor, axis, delta)
    rust_res = rust_f1(tensor, axis, delta)
    assert np.allclose(py_res, rust_res)


def test_take_finite_difference2_rust_matches_python():
    _, rust_f2, _ = _import_rust_funcs()
    tensor = np.random.rand(3, 4, 5)
    delta = [0.1, 0.2, 0.3]
    axis1, axis2 = 0, 2
    py_res = takeFiniteDifference2(tensor, axis1, axis2, delta)
    rust_res = rust_f2(tensor, axis1, axis2, delta)
    assert np.allclose(py_res, rust_res)


def test_ricci_t_loops_rust_matches_python():
    _, _, rust_loop = _import_rust_funcs()
    diff1_flat = np.random.rand(4, 4, 4, 2)
    diff2_flat = np.random.rand(4, 4, 4, 4, 2)
    inv_flat = np.random.rand(4, 4, 2)
    py_res = _ricciT_loops(diff1_flat, diff2_flat, inv_flat)
    rust_res = rust_loop(diff1_flat, diff2_flat, inv_flat)
    assert np.allclose(py_res, rust_res)
