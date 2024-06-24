import numpy as np
from datetime import date

def c4Inv(tensor):
    """
    Computes the inverse of each 4x4 sub-tensor within an n-dimensional tensor.

    Args:
    - tensor (np.ndarray): An n-dimensional tensor where the first two dimensions are 4x4 matrices.

    Returns:
    - inv_tensor (np.ndarray): An n-dimensional tensor of the same shape as the input,
      where each 4x4 sub-tensor has been inverted. If a sub-tensor is singular and cannot
      be inverted, it is replaced by the 4x4 identity matrix.
    """
    # Ensure the tensor has at least 2 dimensions of size 4 for inversion
    if tensor.shape[0] != 4 or tensor.shape[1] != 4:
        raise ValueError("The first two dimensions of the input tensor must be of size 4.")

    # Flatten the remaining dimensions into one dimension
    reshaped_tensor = tensor.reshape(4, 4, -1)

    # Initialize an array to store the inverted sub-tensors
    inv_tensor = np.zeros_like(reshaped_tensor)

    # Loop through the flattened dimension and invert each sub-tensor
    for idx in range(reshaped_tensor.shape[2]):
        sub_tensor = reshaped_tensor[:, :, idx]
        try:
            inv_tensor[:, :, idx] = np.linalg.inv(sub_tensor)
        except np.linalg.LinAlgError:
            inv_tensor[:, :, idx] = np.eye(4)

    # Reshape the inverted tensor back to its original shape
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

    ricci_tensor = np.zeros((4, 4, *s))

    for i in range(4):
        for j in range(i, 4):
            R_munu_temp = np.zeros(s)
            for a in range(4):
                for b in range(4):
                    R_munu_temp -= 0.5 * (diff_2_gl[i, j, a, b] + diff_2_gl[a, b, i, j] - diff_2_gl[i, b, j, a] - diff_2_gl[j, b, i, a]) * inv_metric[a, b]
            for a in range(4):
                for b in range(4):
                    for c in range(4):
                        for d in range(4):
                            R_munu_temp += 0.5 * (0.5 * diff_1_gl[a, c, i] * diff_1_gl[b, d, j] + diff_1_gl[i, c, a] * diff_1_gl[j, d, b] - diff_1_gl[i, c, a] * diff_1_gl[j, b, d]) * inv_metric[a, b] * inv_metric[c, d]
                            R_munu_temp -= 0.25 * (diff_1_gl[j, c, i] + diff_1_gl[i, c, j] - diff_1_gl[i, j, c]) * (2 * diff_1_gl[b, d, a] - diff_1_gl[a, b, d]) * inv_metric[a, b] * inv_metric[c, d]
            ricci_tensor[i, j] = R_munu_temp
            if i != j:
                ricci_tensor[j, i] = R_munu_temp

    return ricci_tensor

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
