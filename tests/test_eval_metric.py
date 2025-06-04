import numpy as np
import pytest
from warp.analyzer.eval_metric import eval_metric

np.random.seed(0)

@pytest.fixture
def setup_metric():
    metric = {
        'type': 'metric',
        'index': 'covariant',
        'tensor': np.random.rand(4, 4, 4, 4),
        'coords': None  # Placeholder for actual coordinates
    }
    return metric

def test_eval_metric(setup_metric):
    metric = setup_metric
    output = eval_metric(metric, try_gpu=0, keep_positive=1, num_angular_vec=100, num_time_vec=10)
    assert 'metric' in output
    assert 'energy_tensor' in output
    assert 'energy_tensor_eulerian' in output
    assert 'null' in output
    assert 'weak' in output
    assert 'strong' in output
    assert 'dominant' in output
    assert 'expansion' in output
    assert 'shear' in output
    assert 'vorticity' in output

def test_eval_metric_no_positive(setup_metric):
    metric = setup_metric
    output = eval_metric(metric, try_gpu=0, keep_positive=0, num_angular_vec=100, num_time_vec=10)
    assert np.all(output['null'] <= 0)
    assert np.all(output['weak'] <= 0)
    assert np.all(output['strong'] <= 0)
    assert np.all(output['dominant'] <= 0)

if __name__ == '__main__':
    pytest.main()
