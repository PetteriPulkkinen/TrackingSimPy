import numpy as np


def basic_measurement_matrix(dim, order):
    """Measurement matrix for position measurements."""
    H = np.zeros(shape=(dim, dim*(order+1)))
    for idx in range(dim):
        H[idx, idx*(order+1)] = 1
    return H


class Sensor(object):
    def __init__(self, H=None, R=None):
        self.R = R
        self.H = H

    def measure(self, x, R=None, H=None):
        if H is None:
            H = self.H
        if R is None:
            R = self.R

        z = (H @ x).flatten()
        return z + np.random.multivariate_normal(np.zeros_like(z), R)

