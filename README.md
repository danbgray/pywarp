# PyWarp

Pywarp is a set of functions and methods to evaluate warp drive spacetimes using Einstein's theory of General Relativity inspired by Warp Factory https://github.com/NerdsWithAttitudes/WarpFactory as a numerical process.

PyWarp is a Python package for calculating the energy tensor from a given metric tensor using Einstein's Field Equations. This package provides a set of functions to compute various tensors, including the Ricci tensor, Ricci scalar, Einstein tensor, and the energy density tensor.

## Installation

To install PyWarp, you can clone the repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/pywarp.git
cd pywarp
pip install -r requirements.txt
```

## Usage

To use PyWarp, you can import the necessary functions and pass the required parameters. Here is an example:
```
import numpy as np
from warp.solver.get_energy_tensor import get_energy_tensor

tensor = np.zeros((4, 4, 2, 2, 2, 2))
for i in range(2):
    for j in range(2):
        for k in range(2):
            for l in range(2):
                tensor[0, 0, i, j, k, l] = -1  # g_tt = -1
                tensor[1, 1, i, j, k, l] = 1   # g_xx = 1
                tensor[2, 2, i, j, k, l] = 1   # g_yy = 1
                tensor[3, 3, i, j, k, l] = 1   # g_zz = 1

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

# pywarp


Using numpy and matrix operations where possible to keep it fast.
starting in primarly python but going to make use of rust or bend where speed is needed.

Goal is to get jupyter notebooks able to explore things
