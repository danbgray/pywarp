import numpy as np
from warp.visualizer import plot_scalar_field, plot_vector_field


def test_plot_scalar_field_matplotlib():
    field = np.random.rand(4, 4, 4)
    fig = plot_scalar_field(field, backend="matplotlib")
    assert hasattr(fig, "canvas")


def test_plot_scalar_field_plotly():
    field = np.random.rand(4, 4, 4)
    fig = plot_scalar_field(field, backend="plotly")
    assert fig.data


def test_plot_vector_field_matplotlib():
    paths = [np.array([[0, 0, 0], [1, 1, 1]])]
    fig = plot_vector_field(paths, backend="matplotlib")
    assert hasattr(fig, "axes")


def test_plot_vector_field_plotly():
    paths = [np.array([[0, 0, 0], [1, 1, 1]])]
    fig = plot_vector_field(paths, backend="plotly")
    assert fig.data
