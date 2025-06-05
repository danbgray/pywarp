# PyWarp

Pywarp is a set of functions and methods to evaluate warp drive spacetimes using Einstein's theory of General Relativity inspired by Warp Factory https://github.com/NerdsWithAttitudes/WarpFactory as a numerical process.

PyWarp is a Python package for calculating the stress-energy tensor from a given metric tensor using Einstein's Field Equations. This package provides a set of functions to compute various tensors, including the Ricci tensor, Ricci scalar, Einstein tensor, and the stress-energy tensor itself.

## Installation

PyWarp uses a Rust extension built with [maturin](https://github.com/PyO3/maturin).
You can either install the package directly with `pip` or set up a development
environment using the helper script.

### Standard `pip` install

Create a fresh virtual environment and install the project. The Rust extension
is compiled automatically during installation.

```bash
python -m venv .venv
source .venv/bin/activate
pip install .
```

### Development setup with `scripts/install.sh`

If you prefer working with `pipenv`, clone the repository and run the helper
script which installs all Python dependencies:

```bash
git clone https://github.com/danbgray/pywarp.git
cd pywarp
./scripts/install.sh
pipenv shell
maturin develop
```
Running `maturin develop` compiles the `warp_core` crate and installs it into
the active environment.

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

print("Energy Tensor shape:", energy['tensor'].shape)
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

The flow line examples demonstrated in the notebook follow the guidelines
outlined in `docs/2102.06824v2.pdf`. After installing the dependencies you can
launch the notebook with:

```bash
jupyter notebook notebooks/intro.ipynb
```

For a full manual that mirrors the structure of the WarpFactory GitBook see
`docs/guide/index.md`.

## Building the Rust extension

The `c4_inv` routine is implemented in Rust for improved performance. Installing
the project with `pip` or running `maturin develop` will compile the extension
for the active environment. To rebuild the extension during development run:

```bash
maturin develop
```
This command compiles the `warp_core` crate and installs it into the current
environment. To create a wheel manually you can also run `maturin build`.

## Testing

The test suite uses `pytest`. After setting up a clean environment and building
the Rust extension you can run the tests. When using `pipenv` the commands are:

```bash
./scripts/install.sh
pipenv shell
maturin develop
pipenv run pytest
```

If you installed the package with `pip` in a virtualenv simply run `pytest` from
that environment.

