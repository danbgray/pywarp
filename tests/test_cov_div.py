import numpy as np
from warp.analyzer.utils import cov_div


def _minkowski_metric(shape):
    g = np.zeros((4, 4) + shape)
    inv = np.zeros_like(g)
    diag = [-1, 1, 1, 1]
    for i in range(4):
        g[i, i, ...] = diag[i]
        inv[i, i, ...] = diag[i]
    return g, inv


def test_cov_div_minkowski_linear_field():
    shape = (3, 3, 3, 3)
    coords = [1, 1, 1, 1]
    t, x, y, z = np.meshgrid(
        np.arange(shape[0]),
        np.arange(shape[1]),
        np.arange(shape[2]),
        np.arange(shape[3]),
        indexing="ij",
    )

    g, inv = _minkowski_metric(shape)

    # u_j = coordinate j
    u_down = [t, x, y, z]
    u_up = [-t, x, y, z]

    result = cov_div(g, inv, u_up, u_down, 1, 1, coords, 0)
    assert np.allclose(result, 1)


def test_cov_div_variable_metric():
    shape = (2, 5, 3, 3)
    coords = [1, 1, 1, 1]
    t, x, y, z = np.meshgrid(
        np.arange(shape[0]),
        np.arange(shape[1]),
        np.arange(shape[2]),
        np.arange(shape[3]),
        indexing="ij",
    )

    g = np.zeros((4, 4) + shape)
    g[0, 0, ...] = -1
    g[1, 1, ...] = 1 + x ** 2
    g[2, 2, ...] = 1
    g[3, 3, ...] = 1

    inv = np.zeros_like(g)
    inv[0, 0, ...] = -1
    inv[1, 1, ...] = 1 / (1 + x ** 2)
    inv[2, 2, ...] = 1
    inv[3, 3, ...] = 1

    u_down = [np.zeros(shape) for _ in range(4)]
    u_down[1] = x
    u_up = [inv[i, i] * u_down[i] for i in range(4)]

    result = cov_div(g, inv, u_up, u_down, 1, 1, coords, 0)
    expected = 1 / (1 + x ** 2)
    # Ignore boundary points where finite difference is one sided
    assert np.allclose(result[:, 1:-1], expected[:, 1:-1])
