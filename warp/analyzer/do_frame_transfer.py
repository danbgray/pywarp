"""Utilities for frame transfers of the stress--energy tensor."""

from typing import Dict, Any

import numpy as np

from warp.solver import verify_tensor
from .utils import (
    change_tensor_index,
    get_eulerian_transformation_matrix,
)

def do_frame_transfer(
    metric: Dict[str, Any],
    energy_tensor: Dict[str, Any],
    frame: str,
    try_gpu: int = 0,
) -> Dict[str, Any]:
    """Convert ``energy_tensor`` to ``frame``.

    Currently only the Eulerian frame is supported.  ``energy_tensor`` is
    expected to be contravariant.  The returned tensor is also
    contravariant and expressed in an orthonormal basis.
    """

    if not verify_tensor(metric):
        raise ValueError(
            "Metric is not verified. Please verify metric using verify_tensor(metric)."
        )
    if not verify_tensor(energy_tensor):
        raise ValueError(
            "Stress-energy is not verified. Please verify stress-energy using verify_tensor(energy_tensor)."
        )

    if frame.lower() != "eulerian":
        raise ValueError("Unsupported frame")

    if energy_tensor.get("frame", "").lower() == "eulerian":
        return energy_tensor

    # Start with a copy so metadata like name/date are preserved
    transformed = energy_tensor.copy()

    # Lower the indices of the energy tensor with the supplied metric
    cov_energy = change_tensor_index(energy_tensor, "covariant", metric)

    # Compute the Eulerian transformation matrix from the metric
    M = get_eulerian_transformation_matrix(metric["tensor"], metric.get("coords"))

    # Ensure the last two dimensions contain the 4x4 matrix for einsum
    cov_array = np.moveaxis(cov_energy["tensor"], [0, 1], [-2, -1])
    M_array = np.moveaxis(M, [0, 1], [-2, -1])

    # Transform to the orthonormal frame: M @ T @ M
    transformed_cov = np.einsum("...ij,...jk,...kl->...il", M_array, cov_array, M_array)
    transformed_cov = np.moveaxis(transformed_cov, [-2, -1], [0, 1])

    # Raise the indices using the Minkowski metric
    minkowski = np.diag([-1.0, 1.0, 1.0, 1.0])
    transformed_contrav = np.einsum(
        "ma,nb,ab...->mn...", minkowski, minkowski, transformed_cov
    )

    transformed.update({
        "tensor": transformed_contrav,
        "frame": "Eulerian",
        "index": "contravariant",
    })

    return transformed
