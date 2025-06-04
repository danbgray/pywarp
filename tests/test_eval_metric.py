import numpy as np
import pytest
from warp.analyzer.eval_metric import eval_metric
from warp.core import minkowski_metric, energy_tensor

np.random.seed(0)

@pytest.fixture
def setup_metric():
    return minkowski_metric((2, 2, 2, 2))

def test_eval_metric(setup_metric):
    metric = setup_metric
    output = eval_metric(metric, try_gpu=0, keep_positive=1, num_angular_vec=10, num_time_vec=5)

    expected_energy = energy_tensor(metric)
    expected_energy['type'] = 'energy'

    assert np.allclose(output['energy_tensor']['tensor'], expected_energy['tensor'])
    assert np.allclose(output['energy_tensor_eulerian']['tensor'], expected_energy['tensor'])
    assert np.all(output['null'] == 0)
    assert np.all(output['weak'] == 0)
    assert np.all(output['strong'] == 0)
    assert np.all(output['dominant'] == 0)
    assert np.all(output['expansion'] == 0)
    assert np.all(output['shear'] == 0)
    assert np.all(output['vorticity'] == 0)

def test_eval_metric_no_positive(setup_metric):
    metric = setup_metric
    output = eval_metric(metric, try_gpu=0, keep_positive=0, num_angular_vec=10, num_time_vec=5)
    assert np.all(output['null'] <= 0)
    assert np.all(output['weak'] <= 0)
    assert np.all(output['strong'] <= 0)
    assert np.all(output['dominant'] <= 0)

if __name__ == '__main__':
    pytest.main()
