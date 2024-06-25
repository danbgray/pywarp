import numpy as np

def setMinkowskiThreePlusOne(gridSize):
    alpha = np.ones(gridSize)
    beta = [np.zeros(gridSize) for _ in range(3)]
    gamma = [np.zeros(gridSize) for _ in range(6)]

    # Assign Minkowski values
    gamma[0][:] = 1  # g_xx = 1
    gamma[1][:] = 1  # g_yy = 1
    gamma[2][:] = 1  # g_zz = 1

    return alpha, beta, gamma

def shapeFunction_Alcubierre(r, R, sigma):
    return np.exp(-((r - R)**2) / sigma**2)

def threePlusOneBuilder(alpha, beta, gamma):
    gridSize = alpha.shape
    tensor = np.zeros((4, 4) + gridSize)

    tensor[0, 0] = -alpha**2
    tensor[1, 1] = gamma[0]
    tensor[2, 2] = gamma[1]
    tensor[3, 3] = gamma[2]

    tensor[0, 1] = tensor[1, 0] = beta[0]
    tensor[0, 2] = tensor[2, 0] = beta[1]
    tensor[0, 3] = tensor[3, 0] = beta[2]

    return tensor
