# PyWarp

Pywarp is a set of functions and methods to evaluate warp drive spacetimes using Einstein's theory of General Relativity inspired by Warp Factory https://github.com/NerdsWithAttitudes/WarpFactory as a numerical process.

PyWarp is a Python package for calculating the energy tensor from a given metric tensor using Einstein's Field Equations. This package provides a set of functions to compute various tensors, including the Ricci tensor, Ricci scalar, Einstein tensor, and the energy density tensor.

## Installation

PyWarp uses a Rust extension built with [maturin](https://github.com/PyO3/maturin).
The extension is compiled automatically when installing the package via `pip`:

```bash
pip install .
```

For development you can still use the helper script which installs the Python
dependencies using `pipenv`:

```bash
git clone https://github.com/yourusername/pywarp.git
cd pywarp
./scripts/install.sh
```

## Usage

To use PyWarp, you can import the necessary functions and pass the required parameters. Here is an example:
```
import numpy as np
from warp.core import energy_tensor

tensor = np.zeros((4, 4, 2, 2, 2, 2))

# Define the values for the diagonal elements
values = [-1, 1, 1, 1]

# Assign values to the diagonal elements for all combinations of the last four indices
for idx in range(4):
    tensor[idx, idx, :, :, :, :] = values[idx]

metric = {
    'type': "Metric",
    'tensor': tensor,
    'coords': "cartesian",
    'index': "covariant",
    'scaling': [1, 1, 1, 1],
    'name': "Minkowski"
}

energy = energy_tensor(metric, diff_order='fourth')

print("Energy Tensor:", energy)
```

For sweeps over multiple metrics you can use `run_parameter_sweep` from the `warp.pipeline` module.
```python
from warp.pipeline import run_parameter_sweep
outputs = run_parameter_sweep([metric])
```


Example notebooks are located in the `notebooks/` directory. Start Jupyter and open
`intro.ipynb` to see a full workflow that builds a metric, computes its energy tensor
and visualizes the result using Plotly. The `rust_demo.ipynb` notebook shows the same
workflow using the Rust extensions for faster linear algebra.

```bash
jupyter notebook notebooks/intro.ipynb
```

## Building the Rust extension

The `c4_inv` routine is implemented in Rust for improved performance. When the
package is installed with `pip` the extension is compiled automatically via the
`pyproject.toml` configuration. If you need to rebuild the extension while
developing, run:

```bash
maturin develop
```

This command compiles the `warp_core` crate and installs it into the current
environment.
