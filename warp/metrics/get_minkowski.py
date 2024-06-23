import numpy as np
from datetime import date

def metric_get_minkowski(grid_size, grid_scaling=(1, 1, 1, 1)):
    """
    Builds a Minkowski metric tensor.

    Args:
    - grid_size (tuple): Size of the world grid in [t, x, y, z].
    - grid_scaling (tuple, optional): Scaling of the grid in [t, x, y, z]. Defaults to (1, 1, 1, 1).

    Returns:
    - metric (dict): Metric tensor represented as a dictionary.
    """
    # Handle default input arguments
    if len(grid_scaling) < 4:
        grid_scaling = (1, 1, 1, 1)

    # Initialize metric dictionary
    metric = {}
    metric['type'] = "metric"
    metric['name'] = "Minkowski"
    metric['scaling'] = grid_scaling
    metric['coords'] = "cartesian"
    metric['index'] = "covariant"
    metric['date'] = date.today()

    # Initialize tensor components
    metric['tensor'] = {
        (1, 1): -np.ones(grid_size),
        (2, 2): np.ones(grid_size),
        (3, 3): np.ones(grid_size),
        (4, 4): np.ones(grid_size),
        (1, 2): np.zeros(grid_size),
        (2, 1): np.zeros(grid_size),
        (1, 3): np.zeros(grid_size),
        (3, 1): np.zeros(grid_size),
        (2, 3): np.zeros(grid_size),
        (3, 2): np.zeros(grid_size),
        (2, 4): np.zeros(grid_size),
        (4, 2): np.zeros(grid_size),
        (3, 4): np.zeros(grid_size),
        (4, 3): np.zeros(grid_size),
        (1, 4): np.zeros(grid_size),
        (4, 1): np.zeros(grid_size)
    }

    return metric
