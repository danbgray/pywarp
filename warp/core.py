"""User-facing API for metric generation and energy tensor calculations."""

from .metrics.get_alcubierre import metric_get_alcubierre, metric_get_alcubierre_comoving
from .metrics.get_minkowski import metric_get_minkowski
from .solver.get_energy_tensor import get_energy_tensor as _get_energy_tensor


def alcubierre_metric(grid_size, world_center, v, R, sigma, grid_scale=(1, 1, 1, 1)):
    """Return the Alcubierre metric dictionary."""
    return metric_get_alcubierre(grid_size, world_center, v, R, sigma, grid_scale)


def alcubierre_comoving_metric(grid_size, world_center, v, R, sigma, grid_scale=(1, 1, 1, 1)):
    """Return the Alcubierre metric in the comoving frame."""
    return metric_get_alcubierre_comoving(grid_size, world_center, v, R, sigma, grid_scale)


def minkowski_metric(grid_size, grid_scaling=(1, 1, 1, 1)):
    """Return a Minkowski metric dictionary."""
    return metric_get_minkowski(grid_size, grid_scaling)


def energy_tensor(metric, diff_order="fourth", try_gpu=0):
    """Compute the stress-energy tensor for *metric*.

    Parameters
    ----------
    metric : dict
        Metric tensor dictionary.
    diff_order : str, optional
        Differentiation order, ``"second"`` or ``"fourth"``.
    try_gpu : int, optional
        Attempt GPU execution when non-zero and CuPy is available.
    """
    return _get_energy_tensor(metric, diffOrder=diff_order, try_gpu=try_gpu)
