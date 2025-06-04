import pytest
from datetime import date
from warp.core import alcubierre_metric, alcubierre_comoving_metric

@pytest.fixture
def setup_params():
    return {
        "gridSize": [10, 10, 10, 10],
        "worldCenter": [5, 5, 5, 5],
        "v": 0.1,
        "R": 2.0,
        "sigma": 1.0,
        "gridScale": [1, 1, 1, 1],
        "c": 1  # speed of light constant
    }

def test_alcubierre_metric(setup_params):
    params = setup_params
    metric = alcubierre_metric(
        params['gridSize'],
        params['worldCenter'],
        params['v'],
        params['R'],
        params['sigma'],
        params['gridScale'],
    )

    assert metric['params']['gridSize'] == params['gridSize']
    assert metric['params']['worldCenter'] == params['worldCenter']
    assert metric['params']['velocity'] == params['v']
    assert metric['params']['R'] == params['R']
    assert metric['params']['sigma'] == params['sigma']
    assert metric['type'] == "metric"
    assert metric['name'] == 'Alcubierre'
    assert metric['scaling'] == params['gridScale']
    assert metric['coords'] == "cartesian"
    assert metric['index'] == "covariant"
    assert metric['date'] == str(date.today())

def test_alcubierre_comoving_metric(setup_params):
    params = setup_params
    params['gridSize'][0] = 1  # Ensure the time grid size is 1 for comoving frame

    metric = alcubierre_comoving_metric(
        params['gridSize'],
        params['worldCenter'],
        params['v'],
        params['R'],
        params['sigma'],
        params['gridScale'],
    )

    assert metric['params']['gridSize'] == params['gridSize']
    assert metric['params']['worldCenter'] == params['worldCenter']
    assert metric['params']['velocity'] == params['v']
    assert metric['params']['R'] == params['R']
    assert metric['params']['sigma'] == params['sigma']
    assert metric['type'] == "metric"
    assert metric['name'] == 'Alcubierre Comoving'
    assert metric['scaling'] == params['gridScale']
    assert metric['coords'] == "cartesian"
    assert metric['index'] == "covariant"
    assert metric['date'] == str(date.today())
