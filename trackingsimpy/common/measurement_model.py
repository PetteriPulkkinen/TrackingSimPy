import numpy as np


def position_measurement_matrix(dim, order):
    """Measurement matrix for position measurements."""
    H = np.zeros(shape=(dim, dim*(order+1)))
    for idx in range(dim):
        H[idx, idx*(order+1)] = 1
    return H
