import numpy as np


def constant_turn_rate_matrix(w, dt):
    """
    f = np.array([
        [1, w**(-1)*np.sin(w*dt), w**(-2)*(1 - np.cos(w * dt))],
        [0, np.cos(w * dt), w**(-1)*np.sin(w*dt)],
        [0, -w*np.sin(w*dt), np.cos(w*dt)]
    ])
    z = np.zeros((3, 3))
    F = np.concatenate((np.concatenate((f, z), axis=0), np.concatenate((z, f), axis=0)), axis=1)
    """

    F = np.array([
        [1, np.sin(w * dt)/w, 0, 0, -(1 - np.cos(w * dt)) / w, 0],
        [0, np.cos(w * dt), 0, 0, -np.sin(w * dt), 0],
        [0, 0, 0, 0, 0, 0],
        [0, (1 - np.cos(w * dt)) / w, 0, 1, np.sin(w * dt) / w, 0],
        [0, np.sin(w * dt), 0, 0, np.cos(w * dt), 0],
        [0, 0, 0, 0, 0, 0],
    ])

    return F


def constant_velocity_matrix(dt):
    F = np.array([
        [1, dt, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, dt],
        [0, 0, 0, 1]
    ])
    return F


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


class DefinedTarget(GenericTarget):
    def __init__(self, x0, st_models, p_noises):
        super(DefinedTarget, self).__init__(x0, None, None)
        self.st_models = st_models
        self.p_noises = p_noises
        self._idx = 0

    def reset(self):
        super().reset()
        self._idx = 0
        self.F = None
        self.Q = None

    def update(self):
        self.F = self.st_models.get(self._idx, self.F)
        self.Q = self.p_noises.get(self._idx, self.Q)
        self._idx += 1
        return super().update()

