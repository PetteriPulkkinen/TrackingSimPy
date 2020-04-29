import numpy as np
from .base_target import BaseTarget


class GenericTargetProcess(BaseTarget):
    def __init__(self, x0, F, Q, order, dim):
        super().__init__(order=order, dim=dim)

        self.x0 = x0.flatten()
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

        self.x = F @ self.x + np.random.multivariate_normal(np.zeros_like(self.x.flatten()), Q).flatten()
        return self.x

    def trajectory_ends(self):
        raise NotImplementedError


