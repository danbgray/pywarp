import numpy as np
import pytest
from warp.analyzer.do_frame_transfer import do_frame_transfer

@pytest.fixture
def setup_tensors():
    metric = {
        'type': 'metric',
        'index': 'covariant',
        'tensor': np.random.rand(4, 4, 4, 4),
        'coords': None  # Placeholder for actual coordinates
    }
    energy_tensor = {
        'type': 'energy',
        'index': 'contravariant',
        'tensor': np.random.rand(4, 4, 4, 4)
    }
    return metric, energy_tensor

def test_do_frame_transfer(setup_tensors):
    metric, energy_tensor = setup_tensors
    transformed_energy_tensor = do_frame_transfer(metric, energy_tensor, 'Eulerian', 0)
    assert transformed_energy_tensor['frame'] == 'Eulerian'
    assert transformed_energy_tensor['index'] == 'contravariant'

def test_invalid_metric(setup_tensors):
    metric, energy_tensor = setup_tensors
    metric['type'] = 'invalid'
    with pytest.raises(ValueError):
        do_frame_transfer(metric, energy_tensor, 'Eulerian', 0)

def test_invalid_energy_tensor(setup_tensors):
    metric, energy_tensor = setup_tensors
    energy_tensor['type'] = 'invalid'
    with pytest.raises(ValueError):
        do_frame_transfer(metric, energy_tensor, 'Eulerian', 0)

def test_unsupported_frame(setup_tensors):
    metric, energy_tensor = setup_tensors
    with pytest.raises(ValueError):
        do_frame_transfer(metric, energy_tensor, 'unsupported_frame', 0)

if __name__ == '__main__':
    pytest.main()
