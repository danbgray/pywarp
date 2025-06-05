"""Utilities for simple time evolution simulations."""

from typing import Callable, Iterable, Dict, Any, List

from warp.analyzer.eval_metric import eval_metric


def run_time_evolution(metric_fn: Callable[[float], Dict[str, Any]], times: Iterable[float]) -> List[Dict[str, Any]]:
    """Generate metrics for each time in ``times`` using ``metric_fn`` and evaluate them.

    Parameters
    ----------
    metric_fn : Callable[[float], dict]
        Function returning a metric dictionary for a given time value.
    times : Iterable[float]
        Sequence of time points to evaluate.

    Returns
    -------
    list of dict
        Results from :func:`eval_metric` for each time step.
    """
    results: List[Dict[str, Any]] = []
    for t in times:
        metric = metric_fn(t)
        results.append(eval_metric(metric))
    return results

