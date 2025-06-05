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
    # Implement the Eulerian transformation matrix logic here
    # Placeholder implementation for example
    return np.eye(4)

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


def _christoffel_symbols(metric_tensor: np.ndarray, inv_metric_tensor: np.ndarray, coords: list) -> np.ndarray:
    """Return the Christoffel symbols for ``metric_tensor``.

    Parameters
    ----------
    metric_tensor : np.ndarray
        Covariant metric tensor of shape ``(4, 4, ...)``.
    inv_metric_tensor : np.ndarray
        Contravariant metric tensor of the same shape.
    coords : list
        Coordinate spacings along each axis.
    """

    shape = metric_tensor.shape[2:]
    partial = [
        np.gradient(metric_tensor, coords[a], axis=2 + a)
        if metric_tensor.shape[2 + a] > 1 else np.zeros_like(metric_tensor)
        for a in range(4)
    ]

    Gamma = np.zeros((4, 4, 4) + shape)
    for a in range(4):
        for b in range(4):
            for c in range(4):
                term = 0
                for d in range(4):
                    term += inv_metric_tensor[a, d] * (
                        partial[b][d, c] + partial[c][d, b] - partial[d][b, c]
                    )
                Gamma[a, b, c] = 0.5 * term
    return Gamma

def cov_div(metric_tensor: np.ndarray, inv_metric_tensor: np.ndarray, u_down: np.ndarray, i: int, j: int, coords: list) -> np.ndarray:
    """Covariant derivative ``∇_j u_i`` for a covariant vector ``u``.

    Parameters
    ----------
    metric_tensor : np.ndarray
        Metric tensor ``g_{ab}``.
    inv_metric_tensor : np.ndarray
        Inverse metric ``g^{ab}``.
    u_down : np.ndarray
        Covariant components of the vector ``u`` with shape ``(4, ...)``.
    i, j : int
        Indices of the derivative.
    coords : list
        Coordinate spacings.
    """

    axis = 1 + j
    if u_down.shape[axis] > 1:
        partial = np.gradient(u_down[i], coords[j], axis=axis)
    else:
        partial = np.zeros_like(u_down[i])

    Gamma = _christoffel_symbols(metric_tensor, inv_metric_tensor, coords)

    result = partial.copy()
    for k in range(4):
        result -= Gamma[k, i, j] * u_down[k]
    return result
