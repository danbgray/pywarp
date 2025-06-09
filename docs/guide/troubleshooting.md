# Troubleshooting

## Singular matrix errors

Some routines invert the metric tensor and will fail if it becomes singular. If you encounter `ValueError: Metric tensor is singular and cannot be inverted`, reduce the bubble velocity or increase `R` and `sigma` so the grid adequately resolves the warp bubble.

## Recommended Alcubierre parameters

For experimentation the notebooks use:

```python
from warp.core import alcubierre_metric
metric = alcubierre_metric((1, 64, 64, 64), (0, 32, 32, 32), v=0.5, R=8, sigma=4)
```

These values avoid singularities on a 64\^3 grid and work well with the examples in [intro.ipynb](../notebooks/intro.ipynb).
