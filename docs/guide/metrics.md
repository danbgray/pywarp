# Available Metrics

PyWarp provides helper functions for common warp-drive metrics. These functions mirror the MATLAB routines in WarpFactory and expose the same parameterization.

| Function | Description |
|----------|-------------|
| `warp.core.alcubierre_metric` | Standard Alcubierre spacetime. |
| `warp.core.alcubierre_comoving_metric` | Alcubierre metric in the comoving frame. |
| `warp.core.minkowski_metric` | Flat Minkowski metric for reference. |

Each function returns a dictionary describing the metric tensor on the simulation grid. The dictionary includes the tensor components, coordinate system, and grid scaling information. See the [notebooks](../notebooks) for concrete examples of constructing metrics.
