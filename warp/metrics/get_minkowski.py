import numpy as np
from datetime import date

def metric_get_minkowski(grid_size, grid_scaling=(1, 1, 1, 1)):
    """
    Builds a Minkowski metric tensor.

    Args:
    - grid_size (tuple): Size of the world grid in (t, x, y, z).
    - grid_scaling (tuple, optional): Scaling of the grid in (t, x, y, z). Defaults to (1, 1, 1, 1).

    Returns:
    - metric (dict): Metric tensor represented as a dictionary.
    """
    # Handle default input arguments
    if len(grid_scaling) < 4:
        grid_scaling = (1, 1, 1, 1)

    # Initialize metric dictionary
    metric = {
        'type': "metric",
        'name': "Minkowski",
        'scaling': grid_scaling,
        'coords': "cartesian",
        'index': "covariant",
        'date': date.today()
    }

    # Initialize tensor components
    tensor = np.zeros((4, 4) + tuple(grid_size))

    # dt^2 term
    tensor[0, 0, ...] = -1

    # Non-time diagonal terms
    for i in range(1, 4):
        tensor[i, i, ...] = 1

    # Cross terms
    for i in range(4):
        for j in range(4):
            if i != j:
                tensor[i, j, ...] = 0

    # Assign tensor to metric
    metric['tensor'] = tensor

    return metric
