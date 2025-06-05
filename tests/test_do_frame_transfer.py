import numpy as np
import pytest

from warp.analyzer.do_frame_transfer import do_frame_transfer
from warp.analyzer.utils import change_tensor_index, get_eulerian_transformation_matrix

np.random.seed(0)

@pytest.fixture
def setup_tensors():
    metric = {
        'type': 'metric',
        'index': 'covariant',
        'tensor': np.random.rand(4, 4, 4, 4),
        'coords': 'cartesian'
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

    # build expected tensor using utilities
    cov_energy = change_tensor_index(energy_tensor, 'covariant', metric)
    M = get_eulerian_transformation_matrix(metric['tensor'], metric.get('coords'))
    cov_array = np.moveaxis(cov_energy['tensor'], [0, 1], [-2, -1])
    M_array = np.moveaxis(M, [0, 1], [-2, -1])
    expected_cov = np.einsum('...ij,...jk,...kl->...il', M_array, cov_array, M_array)
    expected_cov = np.moveaxis(expected_cov, [-2, -1], [0, 1])
    eta = np.diag([-1.0, 1.0, 1.0, 1.0])
    expected_tensor = np.einsum('ma,nb,ab...->mn...', eta, eta, expected_cov)

    assert transformed_energy_tensor['frame'] == 'Eulerian'
    assert transformed_energy_tensor['index'] == 'contravariant'
    assert np.allclose(transformed_energy_tensor['tensor'], expected_tensor)

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
