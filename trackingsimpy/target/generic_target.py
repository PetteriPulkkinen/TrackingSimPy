import numpy as np


class GenericTarget(object):
    def __init__(self, x0, F, Q):
        self.x0 = x0.reshape(-1, 1)
        self.x = self.x0
        self.F = F
        self.Q = Q

    def reset(self):
        self.x = self.x0

    def update(self, F=None, Q=None):
        if F is None:
            F = self.F

        if Q is None:
            Q = self.Q

        self.x = F @ self.x + np.random.multivariate_normal(np.zeros_like(self.x.flatten()), Q).reshape(-1, 1)
        return self.x


