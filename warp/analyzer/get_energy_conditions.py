from typing import Dict, Any, Tuple, Union
import numpy as np
from warp.analyzer.utils import change_tensor_index, generate_uniform_field, get_minkowski_metric, get_trace, verify_tensor, do_frame_transfer, get_inner_product
from warp.metrics.get_minkowski import metric_get_minkowski

def get_energy_conditions(energy_tensor: Dict[str, Any], metric: Dict[str, Any], condition: str, num_angular_vec: int = 100, num_time_vec: int = 10, return_vec: int = 0) -> Tuple[np.ndarray, Union[np.ndarray, None], Union[np.ndarray, None]]:
    """
    Function to get the energy conditions of an energy tensor.

    Args:
        energy_tensor (dict): Energy tensor.
        metric (dict): Metric tensor.
        condition (str): Energy condition to evaluate ('Null', 'Weak', 'Strong', 'Dominant').
        num_angular_vec (int): Number of equally spaced spatial vectors to evaluate.
        num_time_vec (int): Number of equally spaced temporal shells to evaluate.
        return_vec (int): Flag to return all evaluations and their vectors (0=no, 1=yes).

    Returns:
        tuple: map, vec, vectorFieldOut
    """
    # Ensure correct input arguments
    if condition not in {"Null", "Weak", "Strong", "Dominant"}:
        raise ValueError('Incorrect energy condition input, use either: "Null", "Weak", "Dominant", "Strong"')

    if metric['coords'] != 'cartesian':
        raise ValueError('Evaluation not verified for coordinate systems other than Cartesian!')

    if not verify_tensor(metric):
        raise ValueError("Metric is not verified. Please verify metric using verify_tensor(metric).")

    if not verify_tensor(energy_tensor):
        raise ValueError("Stress-energy is not verified. Please verify stress-energy using verify_tensor(energy_tensor).")

    # Get size of the spacetime
    a, b, c, d = metric['tensor'].shape[:4]

    # Convert energy tensor into the local inertial frame if not Eulerian
    energy_tensor = do_frame_transfer(metric, energy_tensor, "Eulerian")

    # Build vector fields
    if condition in {"Null", "Dominant"}:
        field_type = "nulllike"
    elif condition in {"Weak", "Strong"}:
        field_type = "timelike"

    vec_field = generate_uniform_field(field_type, num_angular_vec, num_time_vec)

    # Declare variables to be determined in eval of energy conditions
    map = np.full((a, b, c, d), np.inf)
    vec = np.zeros((a, b, c, d, num_angular_vec, num_time_vec)) if return_vec else None

    if condition == "Null":
        energy_tensor = change_tensor_index(energy_tensor, "covariant", metric)
        for ii in range(num_angular_vec):
            temp = np.einsum('...ij,...i,...j->...', energy_tensor['tensor'], vec_field[:, ii], vec_field[:, ii])
            map = np.minimum(map, temp[..., np.newaxis, np.newaxis, np.newaxis])
            if return_vec:
                vec[:, :, :, :, ii] = temp

    elif condition == "Weak":
        energy_tensor = change_tensor_index(energy_tensor, "covariant", metric)
        for jj in range(num_time_vec):
            for ii in range(num_angular_vec):
                temp = np.einsum('...ij,...i,...j->...', energy_tensor['tensor'], vec_field[:, ii, jj], vec_field[:, ii, jj])
                map = np.minimum(map, temp[..., np.newaxis, np.newaxis, np.newaxis])
                if return_vec:
                    vec[:, :, :, :, ii, jj] = temp

    elif condition == "Dominant":
        metric_minkowski = metric_get_minkowski((a, b, c, d))
        metric_minkowski = change_tensor_index(metric_minkowski, "covariant")

        energy_tensor = change_tensor_index(energy_tensor, "mixedupdown", metric_minkowski)
        for ii in range(num_angular_vec):
            temp = np.einsum('...i,...ij->...j', vec_field[:, ii], energy_tensor['tensor'])
            temp = np.einsum('...i,...i->...', temp, vec_field[:, ii])
            map = np.maximum(map, temp[..., np.newaxis, np.newaxis, np.newaxis])
            if return_vec:
                vec[:, :, :, :, ii] = temp

    elif condition == "Strong":
        metric_minkowski = metric_get_minkowski((a, b, c, d))
        metric_minkowski = change_tensor_index(metric_minkowski, "covariant")

        energy_tensor = change_tensor_index(energy_tensor, "covariant", metric_minkowski)
        E_trace = get_trace(energy_tensor, metric_minkowski)

        for jj in range(num_time_vec):
            for ii in range(num_angular_vec):
                temp = np.einsum('...ij,...i,...j->...', energy_tensor['tensor'] - 0.5 * E_trace[..., np.newaxis, np.newaxis] * metric_minkowski['tensor'], vec_field[:, ii, jj], vec_field[:, ii, jj])
                map = np.minimum(map, temp[..., np.newaxis, np.newaxis, np.newaxis])
                if return_vec:
                    vec[:, :, :, :, ii, jj] = temp

    return map, vec, vec_field if return_vec else None

def get_inner_product(vector1: Dict[str, np.ndarray], vector2: Dict[str, np.ndarray], metric: Dict[str, Any]) -> np.ndarray:
    """
    Calculate the inner product of two vectors using the given metric tensor.

    Args:
        vector1 (dict): First vector.
        vector2 (dict): Second vector.
        metric (dict): Metric tensor.

    Returns:
        np.ndarray: Inner product result.
    """
    return np.einsum('i...,ij...,j...->...', vector1['field'], metric['tensor'], vector2['field'])
