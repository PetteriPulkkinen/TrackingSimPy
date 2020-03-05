import numpy as np


def measurement_matrix(SN0, distance=0, beamwidth=0):
    snr = SN0 * np.exp(- distance / beamwidth)
    alpha = 1 / np.sqrt(snr)
    return np.array([
            [alpha]
    ])


class MonostaticRadar(object):
    def __init__(self, SN0, pf, beamwidth):
        self.H = np.array([
            [1, 0, 0],
        ])
        self.R = lambda dist: measurement_matrix(self.SN0, dist, self.beamwidth)
        self.n = self.H.shape[0]
        self.SN0 = SN0
        self.pf = pf
        self.beamwidth = beamwidth

    def measure(self, x, y):
        """
        :param x: Target state to be measured
        :param y: Target state belief
        :return: Measurement of the target state corrupted with white noise
        """
        distance = np.linalg.norm(x[0] - y[0])
        y = self.H @ x
        return y + np.random.multivariate_normal(np.zeros_like(y), self.R(distance)), self.R(distance)
