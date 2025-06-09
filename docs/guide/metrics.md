# Available Metrics

PyWarp provides helper functions for common warp-drive metrics. These functions mirror the MATLAB routines in WarpFactory and expose the same parameterization.

| Function | Description |
|----------|-------------|
| `warp.core.alcubierre_metric` | Builds the canonical Alcubierre warp metric. |
| `warp.core.alcubierre_comoving_metric` | Alcubierre metric in a comoving frame (time dimension fixed to one). |
| `warp.core.minkowski_metric` | Flat Minkowski metric used for reference. |

Each function returns a dictionary describing the metric tensor on the simulation grid. The dictionary includes the tensor components, coordinate system, and grid scaling information. See the [notebooks](../notebooks) for concrete examples of constructing metrics.

## Parameters

### `alcubierre_metric`
`grid_size` defines the `(t, x, y, z)` dimensions while `world_center` sets the bubble centre in grid units. The arguments `v`, `R` and `sigma` control the bubble velocity, radius and wall thickness respectively. Optional `grid_scale` values convert grid indices to physical distances (see [Units and Conventions](units.md)).

### `alcubierre_comoving_metric`
Accepts the same parameters as `alcubierre_metric` but enforces `grid_size[0] == 1` to produce a single time slice in the comoving frame.

### `minkowski_metric`
Takes a `grid_size` tuple and an optional `grid_scaling` argument to build a flat spacetime for baseline comparisons.
