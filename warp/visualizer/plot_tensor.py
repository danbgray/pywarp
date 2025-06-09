"""Plotting utilities for tensors and fields."""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np

__all__ = ["plot_tensor", "plot_scalar_field", "plot_vector_field"]


def _get_backend(backend: str):
    backend = backend.lower()
    if backend == "matplotlib":
        import matplotlib

        return matplotlib
    if backend == "plotly":
        import plotly

        return plotly
    raise ValueError(f"Unsupported backend: {backend}")


def plot_tensor(
    tensor: np.ndarray,
    *,
    component: Optional[Sequence[int]] = None,
    slice_index: int = 0,
    backend: str = "matplotlib",
    title: Optional[str] = None,
    show: bool = False,
):
    """Plot a tensor component using ``matplotlib`` or ``plotly``.

    Parameters
    ----------
    tensor:
        Array representing the tensor. The last three dimensions are interpreted
        as spatial axes.
    component:
        Optional index tuple selecting the tensor component. If ``None`` and the
        array is 3D, it is treated as a scalar field.
    slice_index:
        Index along the z-axis to plot.
    backend:
        Either ``"matplotlib"`` or ``"plotly"``.
    title:
        Optional plot title.
    show:
        If ``True``, display the plot immediately using the chosen backend.

    Returns
    -------
    object
        The figure instance created by the backend.
    """

    array = np.asarray(tensor)
    if component is not None:
        array = array[tuple(component)]

    if array.ndim != 3:
        raise ValueError("Tensor component must be 3-dimensional")

    data = array[:, :, slice_index]
    mod = _get_backend(backend)

    if mod.__name__ == "matplotlib":
        mod.use("Agg", force=True)
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        im = ax.imshow(data.T, origin="lower")
        if title:
            ax.set_title(title)
        fig.colorbar(im, ax=ax)
        if show:
            plt.show()
        return fig

    if mod.__name__ == "plotly":
        import plotly.graph_objects as go

        fig = go.Figure(data=go.Heatmap(z=data.T))
        if title:
            fig.update_layout(title=title)
        if show:
            fig.show()
        return fig

    raise RuntimeError("Invalid backend")


def plot_scalar_field(
    field: np.ndarray,
    *,
    slice_index: int = 0,
    backend: str = "matplotlib",
    title: Optional[str] = None,
    show: bool = False,
):
    """Convenience wrapper to plot a scalar field."""

    return plot_tensor(
        field,
        component=None,
        slice_index=slice_index,
        backend=backend,
        title=title,
        show=show,
    )


def plot_vector_field(
    paths: Iterable[np.ndarray],
    *,
    backend: str = "matplotlib",
    title: Optional[str] = None,
    show: bool = False,
):
    """Plot flow line paths using the chosen backend."""

    mod = _get_backend(backend)

    if mod.__name__ == "matplotlib":
        mod.use("Agg", force=True)
        import matplotlib.pyplot as plt

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        for path in paths:
            ax.plot(path[:, 0], path[:, 1], path[:, 2])
        if title:
            ax.set_title(title)
        if show:
            plt.show()
        return fig

    if mod.__name__ == "plotly":
        import plotly.graph_objects as go

        fig = go.Figure()
        for path in paths:
            fig.add_trace(
                go.Scatter3d(x=path[:, 0], y=path[:, 1], z=path[:, 2], mode="lines")
            )
        if title:
            fig.update_layout(title=title)
        if show:
            fig.show()
        return fig

    raise RuntimeError("Invalid backend")

