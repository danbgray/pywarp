import numpy as np
from typing import Any, Dict
from warp.solver import verify_tensor

def generate_uniform_field(field_type: str, num_angular_vec: int, num_time_vec: int = 1) -> np.ndarray:
    """
    Generate a uniform field for different conditions.

    Args:
        field_type (str): Type of field to generate. Either "nulllike" or "timelike".
        num_angular_vec (int): Number of equally spaced spatial vectors to generate.
        num_time_vec (int, optional): Number of equally spaced temporal vectors to generate. Default is 1.

    Returns:
        np.ndarray: Generated uniform field.
    """
    if field_type == "nulllike":
        field = get_even_points_on_sphere(num_angular_vec).T
        field = np.vstack((field, np.ones((1, num_angular_vec))))
    elif field_type == "timelike":
        field = np.random.rand(4, num_angular_vec, num_time_vec)
        for i in range(num_time_vec):
            field[0, :, i] = np.cosh(np.linspace(0, 1, num_angular_vec))
            field[1:, :, i] = np.sinh(np.linspace(0, 1, num_angular_vec))
    else:
        raise ValueError('Unknown field type')

    return field

def get_eulerian_transformation_matrix(metric_tensor: np.ndarray, coords: Any) -> np.ndarray:
    """
    Get the Eulerian transformation matrix.

    Args:
        metric_tensor (np.ndarray): Metric tensor.
        coords (Any): Coordinates.

    Returns:
        np.ndarray: Eulerian transformation matrix.
    """
    if coords != "cartesian":
        raise ValueError("Only Cartesian coordinates are supported")

    if metric_tensor.shape[0] != 4 or metric_tensor.shape[1] != 4:
        raise ValueError("metric_tensor must be a 4x4 tensor")

    # Diagonal components of the metric are used to normalise the tetrad
    diag = np.diagonal(metric_tensor, axis1=0, axis2=1)

    m_shape = (4, 4) + metric_tensor.shape[2:]
    transform = np.zeros(m_shape, dtype=metric_tensor.dtype)

    transform[0, 0] = 1.0 / np.sqrt(-diag[0])
    transform[1, 1] = 1.0 / np.sqrt(diag[1])
    transform[2, 2] = 1.0 / np.sqrt(diag[2])
    transform[3, 3] = 1.0 / np.sqrt(diag[3])

    return transform

def get_even_points_on_sphere(num_points: int) -> np.ndarray:
    """
    Get evenly distributed points on a sphere.

    Args:
        num_points (int): Number of points.

    Returns:
        np.ndarray: Evenly distributed points on a sphere.
    """
    indices = np.arange(0, num_points, dtype=float) + 0.5

    phi = np.arccos(1 - 2 * indices / num_points)
    theta = np.pi * (1 + 5**0.5) * indices

    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)

    return np.vstack([x, y, z]).T

def get_inner_product(vector1: Dict[str, Any], vector2: Dict[str, Any], metric: Dict[str, Any]) -> np.ndarray:
    """
    Compute the inner product of two vectors with a given metric.

    Args:
        vector1 (dict): First vector.
        vector2 (dict): Second vector.
        metric (dict): Metric tensor.

    Returns:
        np.ndarray: Inner product of the two vectors.
    """
    return np.einsum('i...,ij...,j...->...', vector1['field'], metric['tensor'], vector2['field'])

def get_trace(tensor: Dict[str, Any], metric: Dict[str, Any]) -> np.ndarray:
    """
    Get the trace of a tensor with the given metric.

    Args:
        tensor (dict): Tensor.
        metric (dict): Metric.

    Returns:
        np.ndarray: Trace of the tensor.
    """
    return np.einsum('...ij,...ij->...', metric['tensor'], tensor['tensor'])

def get_minkowski_metric(shape: list) -> Dict[str, Any]:
    """
    Get the Minkowski metric.

    Args:
        shape (list): Shape of the metric tensor.

    Returns:
        dict: Minkowski metric.
    """
    tensor = np.zeros(shape)
    np.fill_diagonal(tensor, [-1, 1, 1, 1])
    return {'type': 'metric', 'index': 'covariant', 'tensor': tensor}

def do_frame_transfer(metric: Dict[str, Any], energy_tensor: Dict[str, Any], frame: str) -> Dict[str, Any]:
    """
    Perform frame transfer on the energy tensor.

    Args:
        metric (dict): Metric tensor.
        energy_tensor (dict): Energy tensor.
        frame (str): Frame type to transfer to.

    Returns:
        dict: Transformed energy tensor.
    """
    if frame.lower() == 'eulerian':
        transformation_matrix = get_eulerian_transformation_matrix(metric['tensor'], metric.get('coords'))
        transformed_tensor = np.einsum('...ij,...jk,...kl->...il', transformation_matrix, energy_tensor['tensor'], transformation_matrix)
        return {'type': energy_tensor['type'], 'index': energy_tensor['index'], 'tensor': transformed_tensor}
    else:
        raise ValueError('Unsupported frame type')

def change_tensor_index(tensor: Dict[str, Any], index: str, metric: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Change the index of a tensor.

    Args:
        tensor (dict): Tensor to change the index of.
        index (str): New index type.
        metric (dict, optional): Metric tensor. Default is None.

    Returns:
        dict: Tensor with changed index.
    """
    if index == tensor['index']:
        return tensor

    if index == "covariant" and tensor['index'] == "contravariant":
        inv_metric = np.linalg.inv(metric['tensor'])
        new_tensor = np.einsum('...ij,...jk,...kl->...il', inv_metric, tensor['tensor'], inv_metric)
    elif index == "contravariant" and tensor['index'] == "covariant":
        new_tensor = np.einsum('...ij,...jk,...kl->...il', metric['tensor'], tensor['tensor'], metric['tensor'])
    else:
        raise ValueError('Unsupported tensor index transformation')

    return {'type': tensor['type'], 'index': index, 'tensor': new_tensor}

def cov_div(metric_tensor: np.ndarray, inv_metric_tensor: np.ndarray, u_up_cell: np.ndarray, u_down_cell: np.ndarray, i: int, j: int, coords: list, epsilon: float) -> np.ndarray:
    """
    Calculate the covariant derivative of the tensor.

    Args:
        metric_tensor (np.ndarray): Metric tensor.
        inv_metric_tensor (np.ndarray): Inverse of the metric tensor.
        u_up_cell (np.ndarray): Upper indices of the tensor.
        u_down_cell (np.ndarray): Lower indices of the tensor.
        i (int): Index i.
        j (int): Index j.
        coords (list): Coordinates.
        epsilon (float): Small epsilon value.

    Returns:
        np.ndarray: Covariant derivative of the tensor.
    """
    # Placeholder implementation, the actual implementation might be different.
    return np.einsum('...k,...k->...', u_up_cell[i], u_down_cell[j])
