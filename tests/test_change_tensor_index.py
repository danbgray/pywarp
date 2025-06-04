import numpy as np
import pytest
from warp.analyzer.change_tensor_index import change_tensor_index

np.random.seed(0)

@pytest.fixture
def setup_tensors():
    input_tensor = {
        'type': 'normal',
        'index': 'covariant',
        'tensor': np.random.rand(4, 4)
    }
    metric_tensor = {
        'type': 'metric',
        'index': 'covariant',
        'tensor': np.random.rand(4, 4)
    }
    return input_tensor, metric_tensor

def test_covariant_to_contravariant(setup_tensors):
    input_tensor, metric_tensor = setup_tensors
    output_tensor = change_tensor_index(input_tensor, 'contravariant', metric_tensor)
    assert output_tensor['index'] == 'contravariant'

def test_invalid_index(setup_tensors):
    input_tensor, metric_tensor = setup_tensors
    with pytest.raises(ValueError):
        change_tensor_index(input_tensor, 'invalid_index', metric_tensor)

def test_metric_tensor_needed(setup_tensors):
    input_tensor, _ = setup_tensors
    with pytest.raises(ValueError):
        change_tensor_index(input_tensor, 'contravariant')

if __name__ == '__main__':
    pytest.main()
