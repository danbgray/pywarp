import numpy as np
from warp.analyzer.change_tensor_index import change_tensor_index
from warp.analyzer.utils import cov_div, get_trace

def three_plus_one_decomposer(metric):
    shape = metric['tensor'].shape[2:]
    alpha = np.ones(shape)
    beta = np.zeros((3,) + shape)
    gamma = np.zeros((3, 3) + shape)
    beta_up = np.zeros((3,) + shape)
    beta_down = np.zeros((3,) + shape)

    return alpha, beta, gamma, beta_up, beta_down

def get_scalars(metric):
    array_metric_tensor = np.array(metric['tensor'])

    if array_metric_tensor.shape[0:2] != (4, 4):
        raise ValueError("Metric tensor must have shape (4, 4, ...).")

    alpha, _, _, beta_up, _ = three_plus_one_decomposer(metric)

    array_beta = np.array(beta_up)
    s = array_metric_tensor.shape[2:]

    u_up = np.zeros((4, 4) + s)
    u_up[0, 0, ...] = 1 / alpha
    u_up[0, 1:, ...] = -array_beta / alpha
    u_up[1:, 0, ...] = -array_beta / alpha
    u_up[1:, 1:, ...] = np.zeros(array_beta.shape)

    u_down = np.einsum('ij...,j...->i...', array_metric_tensor, u_up)

    u_up_cell = [u_up[i, j, ...] for i in range(4) for j in range(4)]
    u_down_cell = [u_down[i, j, ...] for i in range(4) for j in range(4)]

    metric = change_tensor_index(metric, 'covariant')

    try:
        inv_metric_tensor = np.linalg.inv(metric['tensor'])
    except np.linalg.LinAlgError:
        print(metric['tensor'])
        raise ValueError("Metric tensor is singular and cannot be inverted.")

    del_u_components = np.array([
        cov_div(metric['tensor'], inv_metric_tensor, u_up_cell, u_down_cell, i, j, [1, 1, 1, 1], 0)
        for i in range(4) for j in range(4)
    ])
    del_u_components = del_u_components.reshape(4, 4, *s)

    P_mix = np.eye(4) + np.einsum('...i,...j->...ij', u_up, u_down)
    P = array_metric_tensor + np.einsum('...i,...j->...ij', u_down, u_down)

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
