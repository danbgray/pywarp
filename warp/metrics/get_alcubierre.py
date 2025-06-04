import numpy as np
from datetime import date
from .utils import setMinkowskiThreePlusOne, shapeFunction_Alcubierre, threePlusOneBuilder

def metric_get_alcubierre(gridSize, worldCenter, v, R, sigma, gridScale=(1, 1, 1, 1)):
    metric = {
        "params": {
            "gridSize": gridSize,
            "worldCenter": worldCenter,
            "velocity": v,
            "R": R,
            "sigma": sigma,
        },
        "type": "metric",
        "name": "Alcubierre",
        "scaling": gridScale,
        "coords": "cartesian",
        "index": "covariant",
        "date": str(date.today()),
    }

    alpha, beta, gamma = setMinkowskiThreePlusOne(gridSize)

    t_indices, x_indices, y_indices, z_indices = np.indices(gridSize)

    x = x_indices * gridScale[1] - worldCenter[1]
    y = y_indices * gridScale[2] - worldCenter[2]
    z = z_indices * gridScale[3] - worldCenter[3]
    t = t_indices * gridScale[0] - worldCenter[0]

    xs = t * v * c
    r = np.sqrt((x - xs)**2 + y**2 + z**2)
    fs = shapeFunction_Alcubierre(r, R, sigma)

    beta[0] = -v * fs

    metric["tensor"] = threePlusOneBuilder(alpha, beta, gamma)

    return metric


def metricGet_Alcubierre(*args, **kwargs):
    """Backward compatible wrapper for :func:`metric_get_alcubierre`."""
    return metric_get_alcubierre(*args, **kwargs)

def metric_get_alcubierre_comoving(gridSize, worldCenter, v, R, sigma, gridScale=(1, 1, 1, 1)):
    if gridSize[0] > 1:
        raise ValueError('The time grid is greater than 1, only a size of 1 can be used in comoving')

    metric = {
        "params": {
            "gridSize": gridSize,
            "worldCenter": worldCenter,
            "velocity": v,
            "R": R,
            "sigma": sigma,
        },
        "type": "metric",
        "name": "Alcubierre Comoving",
        "scaling": gridScale,
        "coords": "cartesian",
        "index": "covariant",
        "date": str(date.today()),
    }

    alpha, beta, gamma = setMinkowskiThreePlusOne(gridSize)

    x_indices, y_indices, z_indices = np.indices(gridSize[1:])

    x = x_indices * gridScale[1] - worldCenter[1]
    y = y_indices * gridScale[2] - worldCenter[2]
    z = z_indices * gridScale[3] - worldCenter[3]

    r = np.sqrt(x**2 + y**2 + z**2)
    fs = shapeFunction_Alcubierre(r, R, sigma)

    beta[0] = v * (1 - fs)

    metric["tensor"] = threePlusOneBuilder(alpha, beta, gamma)

    return metric


def metricGet_AlcubierreComoving(*args, **kwargs):
    """Backward compatible wrapper for :func:`metric_get_alcubierre_comoving`."""
    return metric_get_alcubierre_comoving(*args, **kwargs)

# Define the speed of light constant (in whatever units you're using)
c = 1  # You may need to adjust this value based on your unit system
