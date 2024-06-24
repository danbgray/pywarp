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
