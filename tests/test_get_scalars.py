import pytest
from warp.analyzer.get_scalars import get_scalars
from warp.metrics.get_minkowski import metric_get_minkowski

def test_get_scalars_error():
    """Ensure get_scalars raises a ValueError for the default Minkowski metric."""
    grid_size = (2, 2, 2, 2)
    metric = metric_get_minkowski(grid_size)

    with pytest.raises(ValueError):
        get_scalars(metric)

