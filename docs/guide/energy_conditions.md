# Energy Condition Evaluation

Energy conditions quantify how the stress--energy tensor behaves with respect to different classes of observers. PyWarp implements the classic point-wise conditions:

- **Null Energy Condition (NEC)**
- **Weak Energy Condition (WEC)**
- **Dominant Energy Condition (DEC)**
- **Strong Energy Condition (SEC)**

Use `warp.analyzer.get_energy_conditions` or `warp.analyzer.eval_metric` to compute maps of these quantities. The computation follows the same procedure as described in the WarpFactory GitBook and supports GPU acceleration when available.

The resulting arrays have the same spatial shape as the metric and can be visualized directly with `matplotlib` or `plotly`. See the [intro notebook](../notebooks/intro.ipynb) for a demonstration.
