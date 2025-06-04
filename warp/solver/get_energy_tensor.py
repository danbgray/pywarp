import numpy as np
from datetime import date
from numba import njit

try:
    from warp_core import c4_inv as c4Inv, ricci_t_loops as _ricciT_loops_rust

    def _ricciT_loops(diff1_flat, diff2_flat, inv_flat):
        """Call the Rust implementation of the Ricci tensor loops."""
        return _ricciT_loops_rust(diff1_flat, diff2_flat, inv_flat)
except Exception:  # fallback to python implementation
    def c4Inv(tensor):
        """Fallback Python implementation of blockwise 4x4 matrix inversion."""
        if tensor.shape[0] != 4 or tensor.shape[1] != 4:
            raise ValueError("The first two dimensions of the input tensor must be of size 4.")

        reshaped_tensor = tensor.reshape(4, 4, -1)
        inv_tensor = np.zeros_like(reshaped_tensor)
        for idx in range(reshaped_tensor.shape[2]):
            sub_tensor = reshaped_tensor[:, :, idx]
            try:
                inv_tensor[:, :, idx] = np.linalg.inv(sub_tensor)
            except np.linalg.LinAlgError:
                inv_tensor[:, :, idx] = np.eye(4)
        inv_tensor = inv_tensor.reshape(tensor.shape)
        return inv_tensor


def takeFiniteDifference1(tensor, axis, delta):
    """
    Computes the finite difference (gradient) of a tensor along a specified axis.

    Args:
    - tensor (np.ndarray): The input tensor on which to compute the finite difference.
    - axis (int): The axis along which to compute the finite difference.
    - delta (list or np.ndarray): A list or array of spacing values for each axis of the tensor.

    Returns:
    - np.ndarray: The tensor of finite differences (gradients) along the specified axis.
    """
    return np.gradient(tensor, delta[axis], axis=axis)

def takeFiniteDifference2(tensor, axis1, axis2, delta):
    """
    Computes the second-order finite difference (second derivative) of a tensor along two specified axes.

    Args:
    - tensor (np.ndarray): The input tensor on which to compute the second-order finite difference.
    - axis1 (int): The first axis along which to compute the finite difference.
    - axis2 (int): The second axis along which to compute the finite difference.
    - delta (list or np.ndarray): A list or array of spacing values for each axis of the tensor.

    Returns:
    - np.ndarray: The tensor of second-order finite differences (second derivatives) along the specified axes.
    """
    if axis1 == axis2:
        return np.gradient(np.gradient(tensor, delta[axis1], axis=axis1), delta[axis2], axis=axis2)
    else:
        return np.gradient(np.gradient(tensor, delta[axis1], axis=axis1), delta[axis2], axis=axis2)

    @njit
    def _ricciT_loops(diff1_flat, diff2_flat, inv_flat):
        """Numba-accelerated loops for Ricci tensor calculation."""
        ricci_flat = np.zeros((4, 4, diff1_flat.shape[-1]))
        for i in range(4):
            for j in range(i, 4):
                for idx in range(diff1_flat.shape[-1]):
                    temp = 0.0
                    for a in range(4):
                        for b in range(4):
                            temp -= 0.5 * (
                                diff2_flat[i, j, a, b, idx]
                                + diff2_flat[a, b, i, j, idx]
                                - diff2_flat[i, b, j, a, idx]
                                - diff2_flat[j, b, i, a, idx]
                            ) * inv_flat[a, b, idx]
                    for a in range(4):
                        for b in range(4):
                            for c in range(4):
                                for d in range(4):
                                    temp += 0.5 * (
                                        0.5 * diff1_flat[a, c, i, idx] * diff1_flat[b, d, j, idx]
                                        + diff1_flat[i, c, a, idx] * diff1_flat[j, d, b, idx]
                                        - diff1_flat[i, c, a, idx] * diff1_flat[j, b, d, idx]
                                    ) * inv_flat[a, b, idx] * inv_flat[c, d, idx]
                                    temp -= 0.25 * (
                                        diff1_flat[j, c, i, idx]
                                        + diff1_flat[i, c, j, idx]
                                        - diff1_flat[i, j, c, idx]
                                    ) * (
                                        2 * diff1_flat[b, d, a, idx] - diff1_flat[a, b, d, idx]
                                    ) * inv_flat[a, b, idx] * inv_flat[c, d, idx]
                    ricci_flat[i, j, idx] = temp
                    if i != j:
                        ricci_flat[j, i, idx] = temp
        return ricci_flat

def ricciT(inv_metric, metric, delta):
    """
    Computes the Ricci tensor from the given metric tensor and its inverse.

    Args:
    - inv_metric (np.ndarray): The inverse metric tensor.
    - metric (np.ndarray): The metric tensor.
    - delta (list or np.ndarray): A list or array of spacing values for each axis of the tensor.

    Returns:
    - np.ndarray: The computed Ricci tensor.
    """
    s = metric.shape[2:]
    diff_1_gl = np.array([[takeFiniteDifference1(metric[i, j], k, delta) for k in range(4)] for i in range(4) for j in range(4)]).reshape(4, 4, 4, *s)
    diff_2_gl = np.array([[[takeFiniteDifference2(metric[i, j], k, n, delta) for n in range(4)] for k in range(4)] for i in range(4) for j in range(4)]).reshape(4, 4, 4, 4, *s)

    diff1_flat = diff_1_gl.reshape(4, 4, 4, -1)
    diff2_flat = diff_2_gl.reshape(4, 4, 4, 4, -1)
    inv_flat = inv_metric.reshape(4, 4, -1)

    ricci_flat = _ricciT_loops(diff1_flat, diff2_flat, inv_flat)

    return ricci_flat.reshape(4, 4, *s)

def ricciS(ricci_tensor, inv_metric):
    """
    Computes the Ricci scalar from the Ricci tensor and the inverse metric.

    Args:
    - ricci_tensor (np.ndarray): The Ricci tensor.
    - inv_metric (np.ndarray): The inverse metric tensor.

    Returns:
    - np.ndarray: The Ricci scalar.
    """
    ricci_scalar = np.einsum('...ij,...ij', ricci_tensor, inv_metric)
    return ricci_scalar

def einT(ricci_tensor, ricci_scalar, metric):
    """
    Computes the Einstein tensor from the Ricci tensor, Ricci scalar, and metric.

    Args:
    - ricci_tensor (np.ndarray): The Ricci tensor.
    - ricci_scalar (np.ndarray): The Ricci scalar.
    - metric (np.ndarray): The metric tensor.

    Returns:
    - np.ndarray: The Einstein tensor.
    """
    einstein_tensor = ricci_tensor - 0.5 * ricci_scalar[..., np.newaxis, np.newaxis] * metric
    return einstein_tensor

def einE(einstein_tensor, inv_metric):
    """
    Computes the energy density tensor from the Einstein tensor and inverse metric.

    Args:
    - einstein_tensor (np.ndarray): The Einstein tensor.
    - inv_metric (np.ndarray): The inverse metric tensor.

    Returns:
    - np.ndarray: The energy density tensor.
    """
    energy_density = np.einsum('...ij,...ij', einstein_tensor, inv_metric)
    return energy_density

def met2den(tensor, scaling):
    """
    Converts a metric tensor to the corresponding energy density tensor using the Einstein Field Equations.

    Args:
    - tensor (np.ndarray): The metric tensor.
    - scaling (list or np.ndarray): Scaling factors for each axis of the tensor.

    Returns:
    - np.ndarray: The energy density tensor.
    """
    inv_metric = c4Inv(tensor)
    ricci_tensor = ricciT(inv_metric, tensor, scaling)
    ricci_scalar = ricciS(ricci_tensor, inv_metric)
    einstein_tensor = einT(ricci_tensor, ricci_scalar, tensor)
    energy_density = einE(einstein_tensor, inv_metric)
    return energy_density

def met2den2(tensor, scaling):
    """
    Converts a metric tensor to the corresponding energy density tensor using the Einstein Field Equations
    with an alternative method.

    Args:
    - tensor (np.ndarray): The metric tensor.
    - scaling (list or np.ndarray): Scaling factors for each axis of the tensor.

    Returns:
    - np.ndarray: The energy density tensor.
    """
    inv_metric = c4Inv(tensor)
    ricci_tensor = ricciT(inv_metric, tensor, scaling)
    ricci_scalar = ricciS(ricci_tensor, inv_metric)
    einstein_tensor = einT(ricci_tensor, ricci_scalar, tensor)
    energy_density = einE(einstein_tensor, inv_metric)
    return energy_density

def get_energy_tensor(metric, diffOrder='fourth'):
    """
    Computes the energy tensor from a metric tensor using the Einstein Field Equations.

    Args:
    - metric (dict): A dictionary containing the metric tensor and its properties.
    - diffOrder (str, optional): The order of differentiation ('second' or 'fourth'). Defaults to 'fourth'.

    Returns:
    - dict: A dictionary containing the energy tensor and its properties.

    Raises:
    - ValueError: If the diffOrder is not 'second' or 'fourth' or if the metric index is not 'covariant'.
    """
    if diffOrder not in ['second', 'fourth']:
        raise ValueError("Order Flag Not Specified Correctly. Options: 'fourth' or 'second'")

    if metric['index'].lower() != "covariant":
        raise ValueError("Metric index must be 'covariant' for this calculation.")

    if diffOrder == 'fourth':
        energy_tensor = met2den(metric['tensor'], metric['scaling'])
    elif diffOrder == 'second':
        energy_tensor = met2den2(metric['tensor'], metric['scaling'])

    energy = {
        'type': "Stress-Energy",
        'tensor': energy_tensor,
        'coords': metric['coords'],
        'index': "contravariant",
        'order': diffOrder,
        'name': metric['name'],
        'date': date.today()
    }

    return energy
