import numpy as np
from warp.pipeline import run_time_evolution
from warp.metrics.get_minkowski import metric_get_minkowski


def test_run_time_evolution():
    def metric_fn(t):
        return metric_get_minkowski((2, 2, 2, 2))

    times = [0.0, 1.0, 2.0]
    outputs = run_time_evolution(metric_fn, times)
    assert isinstance(outputs, list)
    assert len(outputs) == len(times)
    for out in outputs:
        assert np.allclose(out['energy_tensor']['tensor'], 0)
