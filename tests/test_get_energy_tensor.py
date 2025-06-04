import numpy as np
from warp.solver.get_energy_tensor import get_energy_tensor
from datetime import date

def test_get_energy_tensor():
    """
    Tests the get_energy_tensor function by creating a Minkowski metric tensor and validating
    the resulting energy tensor properties.

    This function creates a 6D tensor representing a Minkowski metric tensor in Cartesian coordinates.
    The tensor is structured to represent the metric at each point in a 4D grid. The function then
    calls get_energy_tensor to compute the energy tensor and validates the result by asserting
    various properties of the output.

    Assertions:
    - The type of the resulting tensor should be "Stress-Energy".
    - The shape of the resulting tensor should be (4, 4, 2, 2).
    - The coordinates of the resulting tensor should be "cartesian".
    - The index type of the resulting tensor should be "contravariant".
    - The order of the computation should be "fourth".
    - The name of the resulting tensor should match the input metric name.
    - The date of the resulting tensor should be a date instance.
    """
    tensor = np.zeros((4, 4, 2, 2, 2, 2))

    # Use broadcasting to set tensor values without loops
    indices = np.arange(2)
    i, j, k, l = np.meshgrid(indices, indices, indices, indices, indexing='ij')

    tensor[0, 0, i, j, k, l] = -1  # g_tt = -1
    tensor[1, 1, i, j, k, l] = 1   # g_xx = 1
    tensor[2, 2, i, j, k, l] = 1   # g_yy = 1
    tensor[3, 3, i, j, k, l] = 1   # g_zz = 1

    metric = {
        'type': "Metric",
        'tensor': tensor,
        'coords': "cartesian",
        'index': "covariant",
        'scaling': [1, 1, 1, 1],
        'name': "Minkowski"
    }

    energy = get_energy_tensor(metric, diffOrder='fourth')

    # Assertions to validate the results
    assert energy['type'] == "Stress-Energy"
    assert energy['tensor'].shape == (4, 4, 2, 2)
    assert energy['coords'] == "cartesian"
    assert energy['index'] == "contravariant"
    assert energy['order'] == 'fourth'
    assert energy['name'] == metric['name']
    assert isinstance(energy['date'], date)

