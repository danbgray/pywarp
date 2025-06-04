import os
import sys
import numpy as np

# Add repository root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from warp.metrics.get_minkowski import metric_get_minkowski
from warp.metrics.get_alcubierre import metricGet_Alcubierre, metricGet_AlcubierreComoving

def test_metric_get_minkowski():
    metric = metric_get_minkowski((1, 1, 1, 1))
    assert metric['type'] == 'metric'
    assert metric['tensor'].shape == (4, 4, 1, 1, 1, 1)
    assert metric['tensor'][0, 0, 0, 0, 0, 0] == -1
    assert metric['tensor'][1, 1, 0, 0, 0, 0] == 1

def test_metricGet_Alcubierre():
    params = {
        'gridSize': [1, 2, 2, 2],
        'worldCenter': [0, 0, 0, 0],
        'v': 0.1,
        'R': 1.0,
        'sigma': 1.0,
        'gridScale': [1, 1, 1, 1]
    }
    metric = metricGet_Alcubierre(
        params['gridSize'],
        params['worldCenter'],
        params['v'],
        params['R'],
        params['sigma'],
        params['gridScale']
    )
    assert metric['type'] == 'metric'
    assert metric['tensor'].shape == (4, 4, 1, 2, 2, 2)


def test_metricGet_AlcubierreComoving():
    params = {
        'gridSize': [1, 2, 2, 2],
        'worldCenter': [0, 0, 0, 0],
        'v': 0.1,
        'R': 1.0,
        'sigma': 1.0,
        'gridScale': [1, 1, 1, 1]
    }
    metric = metricGet_AlcubierreComoving(
        params['gridSize'],
        params['worldCenter'],
        params['v'],
        params['R'],
        params['sigma'],
        params['gridScale']
    )
    assert metric['type'] == 'metric'
    # time grid is 1 so shape preserves that dimension
    assert metric['tensor'].shape == (4, 4, 1, 2, 2, 2)
