import numpy as np
from datetime import date
from numba import njit

try:
    import cupy as cp  # type: ignore
except Exception:  # pragma: no cover - CuPy optional
    cp = None
try:
    from warp_core import (
        c4_inv as c4Inv,
        take_finite_difference1 as rust_take_finite_difference1,
        take_finite_difference2 as rust_take_finite_difference2,
        _ricci_t_loops as rust_ricci_t_loops,
    )
except Exception:  # pragma: no cover - Rust extension optional
    rust_take_finite_difference1 = None
    rust_take_finite_difference2 = None
    rust_ricci_t_loops = None
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


def takeFiniteDifference1(tensor, axis, delta, try_gpu=0):
    """
    Computes the finite difference (gradient) of a tensor along a specified axis.

    Args:
    - tensor (np.ndarray): The input tensor on which to compute the finite difference.
    - axis (int): The axis along which to compute the finite difference.
    - delta (list or np.ndarray): A list or array of spacing values for each axis of the tensor.

    Returns:
    - np.ndarray: The tensor of finite differences (gradients) along the specified axis.
    """
    if try_gpu and cp is not None:
        tensor = cp.asarray(tensor)
        result = cp.gradient(tensor, delta[axis], axis=axis)
        return result
    if rust_take_finite_difference1 is not None and not try_gpu:
        return rust_take_finite_difference1(tensor, axis, delta)
    return np.gradient(tensor, delta[axis], axis=axis)

def takeFiniteDifference2(tensor, axis1, axis2, delta, try_gpu=0):
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
    if try_gpu and cp is not None:
        tensor = cp.asarray(tensor)
        if axis1 == axis2:
            return cp.gradient(cp.gradient(tensor, delta[axis1], axis=axis1), delta[axis2], axis=axis2)
        else:
            return cp.gradient(cp.gradient(tensor, delta[axis1], axis=axis1), delta[axis2], axis=axis2)
    if rust_take_finite_difference2 is not None and not try_gpu:
        return rust_take_finite_difference2(tensor, axis1, axis2, delta)
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


def _ricciT_loops_gpu(diff1_flat, diff2_flat, inv_flat):
    """CuPy implementation of the Ricci tensor loops."""
    ricci_flat = cp.zeros((4, 4, diff1_flat.shape[-1]))
    for i in range(4):
        for j in range(i, 4):
            temp = cp.zeros(diff1_flat.shape[-1])
            for a in range(4):
                for b in range(4):
                    temp -= 0.5 * (
                        diff2_flat[i, j, a, b]
                        + diff2_flat[a, b, i, j]
                        - diff2_flat[i, b, j, a]
                        - diff2_flat[j, b, i, a]
                    ) * inv_flat[a, b]
            for a in range(4):
                for b in range(4):
                    for c in range(4):
                        for d in range(4):
                            temp += 0.5 * (
                                0.5 * diff1_flat[a, c, i] * diff1_flat[b, d, j]
                                + diff1_flat[i, c, a] * diff1_flat[j, d, b]
                                - diff1_flat[i, c, a] * diff1_flat[j, b, d]
                            ) * inv_flat[a, b] * inv_flat[c, d]
                            temp -= 0.25 * (
                                diff1_flat[j, c, i]
                                + diff1_flat[i, c, j]
                                - diff1_flat[i, j, c]
                            ) * (
                                2 * diff1_flat[b, d, a] - diff1_flat[a, b, d]
                            ) * inv_flat[a, b] * inv_flat[c, d]
            ricci_flat[i, j] = temp
            if i != j:
                ricci_flat[j, i] = temp
    return ricci_flat

def ricciT(inv_metric, metric, delta, try_gpu=0):
    """
    Computes the Ricci tensor from the given metric tensor and its inverse.

    Args:
    - inv_metric (np.ndarray): The inverse metric tensor.
    - metric (np.ndarray): The metric tensor.
    - delta (list or np.ndarray): A list or array of spacing values for each axis of the tensor.

    Returns:
    - np.ndarray: The computed Ricci tensor.
    """
    if try_gpu and cp is not None:
        metric = cp.asarray(metric)
        inv_metric = cp.asarray(inv_metric)

    s = metric.shape[2:]
    if try_gpu and cp is not None:
        diff_1_gl = cp.array(
            [[takeFiniteDifference1(metric[i, j], k, delta, try_gpu) for k in range(4)] for i in range(4) for j in range(4)]
        ).reshape(4, 4, 4, *s)
        diff_2_gl = cp.array(
            [[[takeFiniteDifference2(metric[i, j], k, n, delta, try_gpu) for n in range(4)] for k in range(4)] for i in range(4) for j in range(4)]
        ).reshape(4, 4, 4, 4, *s)
    else:
        diff_1_gl = np.array(
            [[takeFiniteDifference1(metric[i, j], k, delta, try_gpu) for k in range(4)] for i in range(4) for j in range(4)]
        ).reshape(4, 4, 4, *s)
        diff_2_gl = np.array(
            [[[takeFiniteDifference2(metric[i, j], k, n, delta, try_gpu) for n in range(4)] for k in range(4)] for i in range(4) for j in range(4)]
        ).reshape(4, 4, 4, 4, *s)

    diff1_flat = diff_1_gl.reshape(4, 4, 4, -1)
    diff2_flat = diff_2_gl.reshape(4, 4, 4, 4, -1)
    inv_flat = inv_metric.reshape(4, 4, -1)

    if try_gpu and cp is not None:
        diff1_flat = cp.asarray(diff1_flat)
        diff2_flat = cp.asarray(diff2_flat)
        inv_flat = cp.asarray(inv_flat)


    if try_gpu and cp is not None:
        ricci_flat = _ricciT_loops_gpu(diff1_flat, diff2_flat, inv_flat)
        ricci_flat = cp.asnumpy(ricci_flat)
    elif rust_ricci_t_loops is not None:
        ricci_flat = rust_ricci_t_loops(diff1_flat, diff2_flat, inv_flat)
    else:
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
    """Raise indices of the Einstein tensor to obtain the stress-energy tensor.

    Parameters
    ----------
    einstein_tensor : np.ndarray
        Einstein tensor with covariant indices.
    inv_metric : np.ndarray
        Inverse metric used to raise the indices.

    Returns
    -------
    np.ndarray
        Contravariant stress-energy tensor with shape ``(4, 4, ...)`` matching
        the spatial dimensions of ``einstein_tensor``.
    """
    return np.einsum("...ij,...ik,...jl->...kl", einstein_tensor, inv_metric, inv_metric)

def met2den(tensor, scaling, try_gpu=0):
    """Convert a metric tensor to its stress-energy tensor.

    Parameters
    ----------
    tensor : np.ndarray
        Metric tensor with covariant indices.
    scaling : list or np.ndarray
        Grid spacing for each axis.

    Returns
    -------
    np.ndarray
        Contravariant stress-energy tensor.
    """
    inv_metric = c4Inv(tensor)
    ricci_tensor = ricciT(inv_metric, tensor, scaling, try_gpu)
    ricci_scalar = ricciS(ricci_tensor, inv_metric)
    einstein_tensor = einT(ricci_tensor, ricci_scalar, tensor)
    return einE(einstein_tensor, inv_metric)

def met2den2(tensor, scaling, try_gpu=0):
    """Alternate method to compute the stress-energy tensor from a metric.

    Parameters
    ----------
    tensor : np.ndarray
        Metric tensor with covariant indices.
    scaling : list or np.ndarray
        Grid spacing for each axis.

    Returns
    -------
    np.ndarray
        Contravariant stress-energy tensor.
    """
    inv_metric = c4Inv(tensor)
    ricci_tensor = ricciT(inv_metric, tensor, scaling, try_gpu)
    ricci_scalar = ricciS(ricci_tensor, inv_metric)
    einstein_tensor = einT(ricci_tensor, ricci_scalar, tensor)
    return einE(einstein_tensor, inv_metric)

def get_energy_tensor(metric, diffOrder='fourth', try_gpu=0):
    """Compute the contravariant stress-energy tensor from ``metric``.

    Parameters
    ----------
    metric : dict
        Metric tensor dictionary with a covariant ``tensor`` entry.
    diffOrder : str, optional
        Differentiation order to use (``'second'`` or ``'fourth'``),
        by default ``'fourth'``.
    try_gpu : int, optional
        Attempt GPU computation when non-zero and CuPy is available.

    Returns
    -------
    dict
        Dictionary containing the stress-energy tensor and metadata.  The tensor
        has shape ``(4, 4, *metric['tensor'].shape[2:])``.

    Raises
    ------
    ValueError
        If ``diffOrder`` is invalid or the metric index is not ``covariant``.
    """
    if diffOrder not in ['second', 'fourth']:
        raise ValueError("Order Flag Not Specified Correctly. Options: 'fourth' or 'second'")

    if metric['index'].lower() != "covariant":
        raise ValueError("Metric index must be 'covariant' for this calculation.")

    if diffOrder == 'fourth':
        energy_tensor = met2den(metric['tensor'], metric['scaling'], try_gpu)
    elif diffOrder == 'second':
        energy_tensor = met2den2(metric['tensor'], metric['scaling'], try_gpu)

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
