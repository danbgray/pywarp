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

    paths = get_momentum_flow_lines(energy_tensor, start_points, step_size, max_steps, scale_factor)
    assert len(paths) == 3

    expected_lengths = [82, 72, 62]
    for path, expected_length in zip(paths, expected_lengths):
        assert path.shape[0] == expected_length
        assert not np.isnan(path).any()


def test_momentum_threshold_stops_integration():
    energy_tensor = np.zeros((4, 4, 4, 4, 4))
    energy_tensor[0, 1] = 0.01
    energy_tensor[0, 2] = 0.01
    energy_tensor[0, 3] = 0.01
    start_points = [np.array([1]), np.array([1]), np.array([1])]
    paths = get_momentum_flow_lines(
        energy_tensor,
        start_points,
        step_size=0.1,
        max_steps=10,
        scale_factor=1.0,
        momentum_threshold=1.0,
    )
    assert paths[0].shape[0] == 1


def test_adaptive_step_normalises_length():
    energy_tensor = np.zeros((4, 4, 4, 4, 4))
    energy_tensor[0, 1] = 0.5
    energy_tensor[0, 2] = 0.0
    energy_tensor[0, 3] = 0.0
    start_points = [np.array([1]), np.array([1]), np.array([1])]
    step_size = 0.2
    paths = get_momentum_flow_lines(
        energy_tensor,
        start_points,
        step_size=step_size,
        max_steps=2,
        scale_factor=1.0,
        adaptive=True,
    )
    delta = paths[0][1] - paths[0][0]
    assert np.isclose(np.linalg.norm(delta), step_size)
