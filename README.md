# PyWarp

Pywarp is a set of functions and methods to evaluate warp drive spacetimes using Einstein's theory of General Relativity inspired by Warp Factory https://github.com/NerdsWithAttitudes/WarpFactory as a numerical process.

PyWarp is a Python package for calculating the energy tensor from a given metric tensor using Einstein's Field Equations. This package provides a set of functions to compute various tensors, including the Ricci tensor, Ricci scalar, Einstein tensor, and the energy density tensor.

## Installation

To install PyWarp, you can clone the repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/pywarp.git
cd pywarp
./scripts/install.sh
```

## Usage

To use PyWarp, you can import the necessary functions and pass the required parameters. Here is an example:
```
import numpy as np
from warp.solver.get_energy_tensor import get_energy_tensor

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

energy = get_energy_tensor(metric, diffOrder='fourth')

print("Energy Tensor:", energy)
```

For sweeps over multiple metrics you can use `run_parameter_sweep` from the `warp.pipeline` module.
```python
from warp.pipeline import run_parameter_sweep
outputs = run_parameter_sweep([metric])
```


Example notebooks are located in the `notebooks/` directory. Start Jupyter and open
`intro.ipynb` to see a full workflow that builds a metric, computes its energy tensor
and visualizes the result using Plotly.

```bash
jupyter notebook notebooks/intro.ipynb
```

## Building the Rust extension

The `c4_inv` routine is implemented in Rust for improved performance. After
installing the Python dependencies run:

```bash
pip install maturin
maturin develop
```

This compiles the `warp_core` crate and makes it available to Python.
