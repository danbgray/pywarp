import pytest
from warp.analyzer.get_scalars import get_scalars
from warp.core import minkowski_metric

def test_get_scalars_error():
    """Ensure get_scalars raises a ValueError for the default Minkowski metric."""
    grid_size = (2, 2, 2, 2)
    metric = minkowski_metric(grid_size)

    with pytest.raises(ValueError):
        get_scalars(metric)

