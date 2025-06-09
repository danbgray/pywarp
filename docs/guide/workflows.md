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
3. Visualize the results with the helpers in ``warp.visualizer``:
   ```python
   from warp.visualizer import plot_scalar_field, plot_vector_field

   # Energy conditions are returned as scalar fields
   fig = plot_scalar_field(results["null"], backend="matplotlib")

   # Momentum flow lines are lists of arrays
   lines = get_momentum_flow_lines(results["energy_tensor"], [...], 0.1, 100, 1.0)
   fig2 = plot_vector_field(lines, backend="plotly")
   ```

Additional examples and visualizations will be added once the pending features from `docs/TASKS.md` are implemented.
