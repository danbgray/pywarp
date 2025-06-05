import numpy as np
import pytest

from warp.analyzer.utils import get_eulerian_transformation_matrix


def test_identity_for_minkowski():
    metric = np.zeros((4, 4))
    np.fill_diagonal(metric, [-1, 1, 1, 1])
    M = get_eulerian_transformation_matrix(metric, "cartesian")
    assert np.allclose(M, np.eye(4))


def test_diagonal_metric():
    diag = [-4.0, 9.0, 16.0, 25.0]
    metric = np.zeros((4, 4))
    np.fill_diagonal(metric, diag)
    M = get_eulerian_transformation_matrix(metric, "cartesian")
    expected = np.diag([0.5, 1/3, 0.25, 0.2])
    assert np.allclose(M, expected)


def test_unsupported_coords():
    metric = np.zeros((4, 4))
    np.fill_diagonal(metric, [-1, 1, 1, 1])
    with pytest.raises(ValueError):
        get_eulerian_transformation_matrix(metric, "spherical")
