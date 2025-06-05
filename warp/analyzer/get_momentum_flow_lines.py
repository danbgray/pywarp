import numpy as np
from scipy.interpolate import RegularGridInterpolator
import warnings



def get_momentum_flow_lines(energy_tensor, start_points, step_size, max_steps, scale_factor):
    """
    Gets the momentum flow lines for an energy tensor.
    Parameters:
    energy_tensor (np.ndarray): Energy tensor (should be contravariant), shape (4, 4, dim_x, dim_y, dim_z).
    start_points (list): List of 3 arrays containing the start points of flowlines.
    step_size (float): Base step size of the flowline propagation.
    max_steps (int): Maximum number of propagation steps to run.
    scale_factor (float): Scaling factor that multiplies the momentum density.
    momentum_threshold (float, optional): Minimum momentum magnitude required
        to continue propagation. Default ``1e-8``.
    adaptive (bool, optional): If ``True`` the step direction is normalised so
        each step has length ``step_size``. Default ``False``.

    Returns:
    list: List of paths, each path is an array of shape (M, 3).
    """
    
    if energy_tensor.shape[0:2] != (4, 4):
        raise ValueError("Energy tensor must have shape (4, 4, dim_x, dim_y, dim_z).")

    # Extract and scale momentum components
    Xmom = energy_tensor[0, 1] * scale_factor
    Ymom = energy_tensor[0, 2] * scale_factor
    Zmom = energy_tensor[0, 3] * scale_factor

    # Define interpolators for momentum components
    grid_x, grid_y, grid_z = (
        np.arange(Xmom.shape[0]),
        np.arange(Xmom.shape[1]),
        np.arange(Xmom.shape[2]),
    )
    interpolator_x = RegularGridInterpolator(
        (grid_x, grid_y, grid_z), Xmom, bounds_error=False, fill_value=np.nan
    )
    interpolator_y = RegularGridInterpolator(
        (grid_x, grid_y, grid_z), Ymom, bounds_error=False, fill_value=np.nan
    )
    interpolator_z = RegularGridInterpolator(
        (grid_x, grid_y, grid_z), Zmom, bounds_error=False, fill_value=np.nan
    )

    # Reshape the starting points
    StrPtsX = np.reshape(start_points[0], -1)
    StrPtsY = np.reshape(start_points[1], -1)
    StrPtsZ = np.reshape(start_points[2], -1)

    paths = []
    all_single_point = True
    for x0, y0, z0 in zip(StrPtsX, StrPtsY, StrPtsZ):
        path = np.zeros((max_steps, 3))
        path[0] = [x0, y0, z0]
        step_count = 1

        for i in range(max_steps - 1):
            pos = path[i]

            if (
                np.any(np.isnan(pos))
                or np.any(pos < 0)
                or np.any(pos >= [Xmom.shape[0], Xmom.shape[1], Xmom.shape[2]])
            ):
                break

            # Interpolate the momentum
            momentum_x = float(interpolator_x(pos))
            momentum_y = float(interpolator_y(pos))
            momentum_z = float(interpolator_z(pos))
            momentum = np.array([momentum_x, momentum_y, momentum_z])

            if np.any(np.isnan(momentum)) or np.allclose(momentum, 0):
                break

            momentum_norm = np.linalg.norm(momentum)
            if momentum_norm < momentum_threshold:
                break

            if adaptive and momentum_norm > 0:
                step_vec = momentum / momentum_norm * step_size
            else:
                step_vec = momentum * step_size

            # Propagate position
            path[i + 1] = pos + step_vec
            step_count += 1

        if step_count > 1:
            all_single_point = False
        paths.append(path[:step_count])

    if all_single_point:
        warnings.warn(
            "All momentum flow lines terminated after a single step. "
            "The energy tensor may contain zero or invalid momentum values.",
            RuntimeWarning,
        )

    return paths


# Example usage:
if __name__ == "__main__":
    # Dummy data for testing
    energy_tensor = np.zeros((4, 4, 10, 10, 10))
    energy_tensor[0, 1, :, :, :] = 1
    energy_tensor[0, 2, :, :, :] = 1
    energy_tensor[0, 3, :, :, :] = 1
    start_points = [np.array([0]), np.array([0]), np.array([0])]
    step_size = 0.1
    max_steps = 100
    scale_factor = 1.0

    paths = get_momentum_flow_lines(
        energy_tensor, start_points, step_size, max_steps, scale_factor
    )
    print(paths)
