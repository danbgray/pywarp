import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from warp.metrics.get_minkowski import metric_get_minkowski
from warp.analyzer.get_energy_conditions import get_energy_conditions


def _setup_zero_energy():
    metric = metric_get_minkowski((1, 1, 1, 1))
    energy_tensor = {
        'tensor': np.zeros_like(metric['tensor']),
        'index': 'covariant',
        'type': 'energy'
    }
    return metric, energy_tensor


def test_null_energy_condition_zero():
    metric, energy = _setup_zero_energy()
    result, _, _ = get_energy_conditions(energy, metric, 'Null', 2, 1, 0)
    assert result.shape[0:2] == (4, 4)
    assert np.allclose(result, 0)


def test_weak_energy_condition_zero():
    metric, energy = _setup_zero_energy()
    result, _, _ = get_energy_conditions(energy, metric, 'Weak', 2, 1, 0)
    assert result.shape[0:2] == (4, 4)
    assert np.allclose(result, 0)


def test_strong_energy_condition_zero():
    metric, energy = _setup_zero_energy()
    result, _, _ = get_energy_conditions(energy, metric, 'Strong', 2, 1, 0)
    assert result.shape[0:2] == (4, 4)
    assert np.allclose(result, 0)


def test_dominant_energy_condition_zero():
    metric, energy = _setup_zero_energy()
    with pytest.raises(ValueError):
        get_energy_conditions(energy, metric, 'Dominant', 2, 1, 0)
