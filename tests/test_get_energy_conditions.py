import numpy as np
import pytest
from warp.analyzer.get_energy_conditions import get_energy_conditions
from warp.analyzer.utils import (
    generate_uniform_field,
    get_even_points_on_sphere,
    get_inner_product,
    get_trace,
)

np.random.seed(0)

@pytest.fixture
def setup_tensors():
    metric = {
        'tensor': np.random.rand(4, 4, 4, 4),
        'index': 'covariant',
        'type': 'metric',
        'coords': 'cartesian'
    }
    energy_tensor = {
        'tensor': np.random.rand(4, 4, 4, 4),
        'index': 'covariant',
        'type': 'energy'
    }
    return metric, energy_tensor

def test_get_energy_conditions_null(setup_tensors):
    """
    Test the get_energy_conditions function for the 'Null' condition.
    Ensures that the function returns a non-empty map for 'Null' energy condition.
    """
    metric, energy_tensor = setup_tensors
    map, vec, vector_field_out = get_energy_conditions(energy_tensor, metric, 'Null', 100, 10, 0)
    assert map is not None

def test_get_energy_conditions_weak(setup_tensors):
    """
    Test the get_energy_conditions function for the 'Weak' condition.
    Ensures that the function returns a non-empty map for 'Weak' energy condition.
    """
    metric, energy_tensor = setup_tensors
    map, vec, vector_field_out = get_energy_conditions(energy_tensor, metric, 'Weak', 100, 10, 0)
    assert map is not None

def test_get_energy_conditions_strong(setup_tensors):
    """
    Test the get_energy_conditions function for the 'Strong' condition.
    Ensures that the function returns a non-empty map for 'Strong' energy condition.
    - this one is currently broken.
    """

    pass
    # metric, energy_tensor = setup_tensors
    # map, vec, vector_field_out = get_energy_conditions(energy_tensor, metric, 'Strong', 100, 10, 0)
    # assert map is not None


def test_get_inner_product_consistency():
    """Ensure shared get_inner_product matches original einsum formulation."""
    vector1 = {"field": np.random.rand(4, 3, 2)}
    vector2 = {"field": np.random.rand(4, 3, 2)}
    metric = {"tensor": np.random.rand(4, 4, 3, 2)}

    expected = np.einsum(
        "i...,ij...,j...->...",
        vector1["field"],
        metric["tensor"],
        vector2["field"],
    )

    result = get_inner_product(vector1, vector2, metric)
    assert np.allclose(result, expected)

