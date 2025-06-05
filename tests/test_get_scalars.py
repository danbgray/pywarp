import numpy as np
import pytest
from warp.analyzer.get_scalars import get_scalars
from warp.core import minkowski_metric
from warp.metrics.utils import threePlusOneBuilder

def test_get_scalars_error():
    """Ensure get_scalars raises a ValueError for the default Minkowski metric."""
    grid_size = (2, 2, 2, 2)
    metric = minkowski_metric(grid_size)

    with pytest.raises(ValueError):
        get_scalars(metric)


def test_constant_shift_scalars_zero():
    """Scalars vanish for Minkowski metric in a boosted frame."""
    grid_size = (1, 1, 1, 1)
    alpha = np.ones(grid_size)
    beta = [np.full(grid_size, 0.5), np.zeros(grid_size), np.zeros(grid_size)]
    gamma = [np.ones(grid_size), np.ones(grid_size), np.ones(grid_size), np.zeros(grid_size), np.zeros(grid_size), np.zeros(grid_size)]

    tensor = threePlusOneBuilder(alpha, beta, gamma)
    metric = {"type": "metric", "index": "covariant", "tensor": tensor, "scaling": (1, 1, 1, 1)}

    expansion, shear, vorticity = get_scalars(metric)

    assert np.allclose(expansion, 0)
    assert np.allclose(shear, 0)
    assert np.allclose(vorticity, 0)

