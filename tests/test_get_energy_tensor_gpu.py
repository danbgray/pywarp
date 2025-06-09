import numpy as np
import pytest

cupy = pytest.importorskip("cupy")  # skip entire module if CuPy not available

from warp.core import minkowski_metric, energy_tensor


def test_energy_tensor_gpu_matches_cpu():
    metric = minkowski_metric((2, 2, 2, 2))
    cpu_res = energy_tensor(metric, try_gpu=0)
    gpu_res = energy_tensor(metric, try_gpu=1)
    assert np.allclose(cpu_res["tensor"], gpu_res["tensor"])

