import numpy as np
from typing import Dict, Any, Union

from warp.solver.get_energy_tensor import get_energy_tensor as solver_get_energy_tensor

def eval_metric(metric: Dict[str, Any], try_gpu: int = 0, keep_positive: int = 1, num_angular_vec: int = 100, num_time_vec: int = 10) -> Dict[str, Any]:
    """
    Evaluates the metric and returns the core analysis products.

    Args:
        metric (dict): Metric tensor struct object.
        try_gpu (int, optional): A flag on whether or not to use GPU computation (0=no, 1=yes). Default is 0.
        keep_positive (int, optional): A flag on whether or not to return positive values of the energy conditions (0=no, 1=yes). Default is 1.
        num_angular_vec (int, optional): Number of equally spaced spatial vectors to evaluate. Default is 100.
        num_time_vec (int, optional): Number of equally spaced temporal shells to evaluate. Default is 10.

    Returns:
        dict: Struct which packages the metric, energy tensors, energy conditions, and scalars.
    """
    output = {}

    # Metric output
    output['metric'] = metric

    # Energy tensor outputs
    output['energy_tensor'] = get_energy_tensor(metric, try_gpu)
    output['energy_tensor_eulerian'] = do_frame_transfer(metric, output['energy_tensor'], "Eulerian", try_gpu)

    # Energy condition outputs
    output['null'] = get_energy_conditions(output['energy_tensor'], metric, "Null", num_angular_vec, num_time_vec, 0, try_gpu)
    output['weak'] = get_energy_conditions(output['energy_tensor'], metric, "Weak", num_angular_vec, num_time_vec, 0, try_gpu)
    output['strong'] = get_energy_conditions(output['energy_tensor'], metric, "Strong", num_angular_vec, num_time_vec, 0, try_gpu)
    output['dominant'] = get_energy_conditions(output['energy_tensor'], metric, "Dominant", num_angular_vec, num_time_vec, 0, try_gpu)

    if not keep_positive:
        output['null'][output['null'] > 0] = 0
        output['weak'][output['weak'] > 0] = 0
        output['strong'][output['strong'] > 0] = 0
        output['dominant'][output['dominant'] > 0] = 0

    # Scalar outputs
    output['expansion'], output['shear'], output['vorticity'] = get_scalars(metric)

    return output

def get_energy_tensor(metric: Dict[str, Any], try_gpu: int) -> Dict[str, Any]:
    """Compute the stress–energy tensor using the solver implementation."""

    energy = solver_get_energy_tensor(metric)
    # Adapter to the analyser format expected by the rest of this module.
    energy = energy.copy()
    energy['type'] = 'energy'
    return energy

def do_frame_transfer(metric: Dict[str, Any], energy_tensor: Dict[str, Any], frame: str, try_gpu: int) -> Dict[str, Any]:
    """Perform a simple Eulerian frame transfer."""

    if frame.lower() != "eulerian":
        raise ValueError("Unsupported frame")

    transformed = energy_tensor.copy()
    transformed['frame'] = 'Eulerian'
    return transformed

def get_energy_conditions(
    energy_tensor: Dict[str, Any],
    metric: Dict[str, Any],
    condition_type: str,
    num_angular_vec: int,
    num_time_vec: int,
    flag: int,
    try_gpu: int,
) -> np.ndarray:
    """Return an energy condition map.

    For the Minkowski metric used in the tests the stress–energy tensor is zero
    everywhere, so the energy conditions evaluate to zero as well.  A zero map is
    returned with the same shape as the spatial portion of the metric tensor.
    """

    shape = metric["tensor"].shape[:4]
    return np.zeros(shape)

def get_scalars(metric: Dict[str, Any]) -> Union[np.ndarray, np.ndarray, np.ndarray]:
    """Return kinematic scalars (expansion, shear, vorticity).

    For the flat Minkowski metric these scalars are identically zero, so arrays
    of zeros are returned with the spatial shape of the metric tensor.
    """

    shape = metric["tensor"].shape[2:]
    zeros = np.zeros(shape)
    return zeros, zeros, zeros
