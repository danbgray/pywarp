# PyWarp Guide

This guide introduces PyWarp usage and mirrors the layout of the original WarpFactory GitBook so users familiar with WarpFactory can transition easily.

## Installation

PyWarp ships a small Rust extension compiled with [maturin](https://github.com/PyO3/maturin). You can install the package directly using `pip` or set up a development environment via the helper script.

### Using `pip`

```bash
python -m venv .venv
source .venv/bin/activate
pip install .
```

### Development setup

Clone the repository and let `scripts/install.sh` install the dependencies. Afterwards build the extension in place.

```bash
git clone https://github.com/yourusername/pywarp.git
cd pywarp
./scripts/install.sh
pipenv shell
maturin develop
```

## Available metrics

`warp.metrics` currently provides generators for a few basic spacetimes:

- `metric_get_minkowski` – flat Minkowski spacetime.
- `metric_get_alcubierre` – the standard Alcubierre warp metric.
- `metric_get_alcubierre_comoving` – a comoving variant of the Alcubierre metric.

These helpers return metric dictionaries ready for further analysis or energy condition evaluation.

## Energy-condition evaluation

Energy conditions can be computed with `warp.analyzer.get_energy_conditions.get_energy_conditions`. Pass a stress–energy tensor and the metric used to generate it along with the condition name ("Null", "Weak", "Strong" or "Dominant"). The routine returns maps showing where a given condition is satisfied.

## Example workflows

Typical analyses create a metric, compute the energy tensor and then check energy conditions. The [notebooks](../notebooks) walk through these steps in detail. Start with [`intro.ipynb`](../notebooks/intro.ipynb) for a basic demonstration or [`rust_demo.ipynb`](../notebooks/rust_demo.ipynb) to see the Rust accelerated functions.

Pipeline helpers such as `warp.pipeline.run_parameter_sweep` simplify evaluating many metrics or evolving a metric over time. More features will be documented here as the stubs in `docs/TASKS.md` are implemented.

