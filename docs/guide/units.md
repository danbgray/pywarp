# Units and Conventions

PyWarp operates in natural units where the speed of light \(c\) is set to one. Time and distance are therefore measured relative to the grid scaling provided when a metric is generated.

## Grid scaling

Each metric generator accepts a `grid_scale` (or `grid_scaling`) argument specifying the spacing along `(t, x, y, z)`. For example `grid_scale=(1, 10, 10, 10)` treats each spatial step as ten times larger than the default.

## Converting to physical units

To convert results to metres and seconds multiply the grid indices by your chosen spatial step and divide or multiply by `c` as required. The constant is defined in `warp.metrics.get_alcubierre` and can be adjusted if you prefer another unit system.

Typical notebooks assume unit spacing which keeps the equations simple. When experimenting with physical scales, ensure that all metric parameters and plotted distances use the same conversion factor.
