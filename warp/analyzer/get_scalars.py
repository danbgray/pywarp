import numpy as np
from warp.analyzer.change_tensor_index import change_tensor_index
from warp.analyzer.utils import cov_div, get_trace


def three_plus_one_decomposer(metric):
    """Return the 3+1 decomposition of ``metric``.

    Parameters
    ----------
    metric : dict
        Metric dictionary with a covariant tensor of shape ``(4, 4, ...)``.

    Returns
    -------
    tuple
        ``alpha, beta, gamma, beta_up, beta_down`` where ``alpha`` is the lapse
        function, ``beta`` is the covariant shift vector, ``gamma`` is the
        spatial metric and ``beta_up``/``beta_down`` are the contravariant and
        covariant versions of the shift vector.
    """

    if metric["index"].lower() != "covariant":
        raise ValueError("Metric must be in covariant index for decomposition")

    g = np.array(metric["tensor"], dtype=float)
    shape = g.shape[2:]

    # Inverse metric at every grid cell
    flat = g.reshape(4, 4, -1)
    inv_flat = np.empty_like(flat)
    for idx in range(flat.shape[-1]):
        inv_flat[:, :, idx] = np.linalg.inv(flat[:, :, idx])
    g_inv = inv_flat.reshape(4, 4, *shape)

    gamma = g[1:, 1:, ...]
    beta_down = g[0, 1:, ...]

    # Inverse of the spatial metric
    flat_gamma = gamma.reshape(3, 3, -1)
    inv_flat_gamma = np.empty_like(flat_gamma)
    for idx in range(flat_gamma.shape[-1]):
        inv_flat_gamma[:, :, idx] = np.linalg.inv(flat_gamma[:, :, idx])
    gamma_inv = inv_flat_gamma.reshape(3, 3, *shape)

    alpha = 1.0 / np.sqrt(-g_inv[0, 0])
    beta_up = np.einsum("ij...,j...->i...", gamma_inv, beta_down)

    beta = beta_down

    return alpha, beta, gamma, beta_up, beta_down


def _invert_metric(tensor: np.ndarray) -> np.ndarray:
    """Return the matrix inverse of ``tensor`` at each grid point."""
    shape = tensor.shape[2:]
    flat = tensor.reshape(4, 4, -1)
    inv_flat = np.empty_like(flat)
    for idx in range(flat.shape[-1]):
        inv_flat[:, :, idx] = np.linalg.inv(flat[:, :, idx])
    return inv_flat.reshape(4, 4, *shape)

def get_scalars(metric):
    """Return kinematic scalars for ``metric``.

    The function is undefined for the flat Minkowski metric.  If the supplied
    metric tensor matches the standard Minkowski tensor a ``ValueError`` is
    raised.
    """

    array_metric_tensor = np.array(metric['tensor'])

    if array_metric_tensor.shape[0:2] != (4, 4):
        raise ValueError("Metric tensor must have shape (4, 4, ...).")

    expected = np.zeros_like(array_metric_tensor)
    expected[0, 0, ...] = -1
    expected[1, 1, ...] = 1
    expected[2, 2, ...] = 1
    expected[3, 3, ...] = 1

    if np.allclose(array_metric_tensor, expected):
        raise ValueError("Minkowski metric provided; scalars are undefined.")

    alpha, beta_down, _, beta_up, _ = three_plus_one_decomposer(metric)

    s = array_metric_tensor.shape[2:]

    u_up = np.zeros((4,) + s)
    u_up[0] = 1.0 / alpha
    u_up[1:] = -beta_up / alpha

    u_down = np.einsum('ij...,j...->i...', array_metric_tensor, u_up)

    metric = change_tensor_index(metric, 'covariant')
    try:
        inv_metric_tensor = _invert_metric(metric['tensor'])
    except np.linalg.LinAlgError:
        raise ValueError("Metric tensor is singular and cannot be inverted.")

    coords = metric.get('scaling', [1, 1, 1, 1])

    del_u_components = np.array([
        cov_div(metric['tensor'], inv_metric_tensor, u_up, u_down, i, j, coords)
        for i in range(4) for j in range(4)
    ]).reshape(4, 4, *s)

    eye = np.eye(4)
    P_mix = eye[:, :, None] + np.einsum('i...,j...->ij...', u_up, u_down)
    P = array_metric_tensor + np.einsum('i...,j...->ij...', u_down, u_down)

    theta = {'index': "covariant", 'type': "tensor", 'tensor': np.zeros((4, 4) + s)}
    omega = {'index': "covariant", 'type': "tensor", 'tensor': np.zeros((4, 4) + s)}

    theta['tensor'] = 0.5 * (np.einsum('...ia,...jb,...ab->...ij', P_mix, P_mix, del_u_components) +
                             np.einsum('...ia,...jb,...ba->...ij', P_mix, P_mix, del_u_components))

    omega['tensor'] = 0.5 * (np.einsum('...ia,...jb,...ab->...ij', P_mix, P_mix, del_u_components) -
                             np.einsum('...ia,...jb,...ba->...ij', P_mix, P_mix, del_u_components))

    theta_trace = get_trace(theta, metric)

    omega_up = change_tensor_index(omega, "contravariant", metric)
    omega_trace = 0.5 * np.einsum('...ij,...ij->...', omega_up['tensor'], omega['tensor'])

    shear = {'index': "covariant", 'type': "tensor", 'tensor': np.zeros((4, 4) + s)}
    shear['tensor'] = theta['tensor'] - theta_trace[..., np.newaxis, np.newaxis] / 3 * P

    shear_up = change_tensor_index(shear, "contravariant", metric)
    sigma2 = 0.5 * np.einsum('...ij,...ij->...', shear['tensor'], shear_up['tensor'])

    shear_scalar = sigma2
    expansion_scalar = theta_trace
    vorticity_scalar = omega_trace

    return expansion_scalar, shear_scalar, vorticity_scalar
