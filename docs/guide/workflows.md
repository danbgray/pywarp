# Example Workflows

The notebooks in the repository showcase how to combine metric generation, stress--energy computation and visualization into a single workflow. Start with `notebooks/intro.ipynb` for a pure Python example or `notebooks/rust_demo.ipynb` to see the Rust extension in action.

Typical steps:

1. Build a metric using one of the helper functions:
   ```python
   from warp.core import alcubierre_metric
   metric = alcubierre_metric((1, 64, 64, 64), (0, 32, 32, 32), v=0.5, R=8, sigma=4)
   ```
2. Compute the stress--energy tensor and evaluate energy conditions:
   ```python
   from warp.core import energy_tensor
   from warp.analyzer import eval_metric
   tensor = energy_tensor(metric)
   results = eval_metric(metric)
   ```
3. Visualize the results with your preferred plotting library.

Additional examples and visualizations will be added once the pending features from `docs/TASKS.md` are implemented.
