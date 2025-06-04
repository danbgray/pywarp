import numpy as np
from warp.pipeline.simulation import run_parameter_sweep
from warp.metrics.get_minkowski import metric_get_minkowski
from warp.analyzer.eval_metric import eval_metric


def test_run_parameter_sweep():
    metrics = [metric_get_minkowski((2, 2, 2, 2)) for _ in range(2)]
    outputs = run_parameter_sweep(metrics)
    assert isinstance(outputs, list)
    assert len(outputs) == 2

    expected = [eval_metric(m) for m in metrics]
    for out, exp in zip(outputs, expected):
        assert np.allclose(out['energy_tensor']['tensor'], exp['energy_tensor']['tensor'])
