"""Simple simulation utilities for batch metric analysis."""

from typing import Iterable, Dict, Any, List

from warp.analyzer.eval_metric import eval_metric


def run_parameter_sweep(metrics: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Evaluate a sequence of metrics using :func:`eval_metric`.

    Parameters
    ----------
    metrics : Iterable[dict]
        Collection of metric dictionaries to evaluate.

    Returns
    -------
    list of dict
        Results from :func:`eval_metric` for each metric in ``metrics``.
    """
    results: List[Dict[str, Any]] = []
    for metric in metrics:
        results.append(eval_metric(metric))
    return results
