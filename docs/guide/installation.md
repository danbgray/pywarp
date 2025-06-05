# Installation

PyWarp uses a Rust extension built with [maturin](https://github.com/PyO3/maturin). The package can be installed directly from the repository. A typical setup mirrors the WarpFactory installation flow but uses Python tooling.

```bash
pip install .
```

For development you can clone the repository and run the helper script which installs Python dependencies with `pipenv`:

```bash
git clone https://github.com/yourusername/pywarp.git
cd pywarp
./scripts/install.sh
```

The Rust extension will compile automatically as part of the installation process. If you need to rebuild it manually, run:

```bash
maturin develop
```
