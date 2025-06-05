import numpy as np
import pytest
from warp.analyzer.get_momentum_flow_lines import get_momentum_flow_lines


def test_get_momentum_flow_lines():
    # Dummy data for testing
    energy_tensor = np.zeros((4, 4, 10, 10, 10))
    energy_tensor[0, 1, :, :, :] = 1
    energy_tensor[0, 2, :, :, :] = 1
    energy_tensor[0, 3, :, :, :] = 1
    start_points = [np.array([1, 2, 3]), np.array([1, 2, 3]), np.array([1, 2, 3])]
    step_size = 0.1
    max_steps = 100
    scale_factor = 1.0

    paths = get_momentum_flow_lines(
        energy_tensor, start_points, step_size, max_steps, scale_factor
    )
    assert len(paths) == 3

    expected_lengths = [82, 72, 62]
    for path, expected_length in zip(paths, expected_lengths):
        assert path.shape[0] == expected_length
        assert not np.isnan(path).any()


def test_zero_momentum_warning():
    energy_tensor = np.zeros((4, 4, 5, 5, 5))
    start_points = [np.array([0]), np.array([0]), np.array([0])]
    with pytest.warns(RuntimeWarning):
        paths = get_momentum_flow_lines(energy_tensor, start_points, 0.1, 10, 1.0)
    assert paths[0].shape[0] == 1
