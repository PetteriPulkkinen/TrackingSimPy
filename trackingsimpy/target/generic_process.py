import numpy as np
from .base_target import BaseTarget


class GenericTargetProcess(BaseTarget):
    def __init__(self, x0, F, Q, max_steps, order, dim,):
        super().__init__(order=order, dim=dim)

        self.x0 = x0.flatten()
        self.x = self.x0.copy()
        self.F = F
        self.Q = Q
        self.max_steps = max_steps

        self._idx = 0

    def reset(self):
        self.x = self.x0.copy()
        self._idx = 0

    def update(self):
        self._idx += 1
        self.x = self.F @ self.x + self.__T @ np.random.randn(self.x.size)
        return self.x

    @property
    def Q(self):
        return self.__Q

    @Q.setter
    def Q(self, Q):
        values, vectors = np.linalg.eig(Q)
        self.__T = vectors @ np.diag(np.sqrt(values))
        self.__Q = Q

    def trajectory_ends(self):
        return self._idx >= self.max_steps


