import numpy as np


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

