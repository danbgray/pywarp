import numpy as np
from datetime import date
from warp.analyzer.get_scalars import get_scalars
from warp.metrics.get_minkowski import metric_get_minkowski

def test_get_scalars():
    grid_size = (2, 2, 2)  # Use a smaller grid size for simplicity
    metric = metric_get_minkowski(grid_size)

    # Print the full metric tensor and its shape to verify its values
    print("Metric Tensor Shape:", metric['tensor'].shape)
    print("Metric Tensor:")
    print(metric['tensor'])

    try:
        expansion_scalar, shear_scalar, vorticity_scalar = get_scalars(metric)
        print("Expansion Scalar:", expansion_scalar)
        print("Shear Scalar:", shear_scalar)
        print("Vorticity Scalar:", vorticity_scalar)
    except ValueError as e:
        print("Error during tensor inversion:", e)

if __name__ == "__main__":
    test_get_scalars()
