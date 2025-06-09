import numpy as np
from typing import Dict, Any

from warp.solver.get_energy_tensor import get_energy_tensor
from warp.analyzer.do_frame_transfer import do_frame_transfer
from warp.analyzer.get_energy_conditions import get_energy_conditions
from warp.analyzer.get_scalars import get_scalars

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
    output['energy_tensor'] = get_energy_tensor(metric, try_gpu=try_gpu)
    output['energy_tensor']['type'] = 'energy'
    output['energy_tensor_eulerian'] = do_frame_transfer(metric, output['energy_tensor'], "Eulerian", try_gpu)

    # Energy condition outputs
    try:
        output['null'], _, _ = get_energy_conditions(
            output['energy_tensor'], metric, "Null", num_angular_vec, num_time_vec, 0
        )
        output['weak'], _, _ = get_energy_conditions(
            output['energy_tensor'], metric, "Weak", num_angular_vec, num_time_vec, 0
        )
        output['strong'], _, _ = get_energy_conditions(
            output['energy_tensor'], metric, "Strong", num_angular_vec, num_time_vec, 0
        )
        output['dominant'], _, _ = get_energy_conditions(
            output['energy_tensor'], metric, "Dominant", num_angular_vec, num_time_vec, 0
        )
    except Exception:
        shape = metric['tensor'].shape[:4]
        output['null'] = np.zeros(shape)
        output['weak'] = np.zeros(shape)
        output['strong'] = np.zeros(shape)
        output['dominant'] = np.zeros(shape)

    if not keep_positive:
        output['null'][output['null'] > 0] = 0
        output['weak'][output['weak'] > 0] = 0
        output['strong'][output['strong'] > 0] = 0
        output['dominant'][output['dominant'] > 0] = 0

    # Scalar outputs
    try:
        expansion, shear, vorticity = get_scalars(metric)
    except ValueError:
        shape = metric['tensor'].shape[2:]
        expansion = np.zeros(shape)
        shear = np.zeros(shape)
        vorticity = np.zeros(shape)

    output['expansion'] = expansion
    output['shear'] = shear
    output['vorticity'] = vorticity

    return output

