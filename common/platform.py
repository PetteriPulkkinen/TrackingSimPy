import numpy as np


class Platform(object):
    def __init__(self, dim):
        self.dim = self.dim
        self.position = np.zeros(dim)
        self.velocity = np.zeros(dim)
        self.acceleration = np.zeros(dim)

    def reset(self):
        pass

    def update(self):
        pass

    def state(self):
        return np.array([
            self.platform.position,
            self.platform.velocity,
            self.platform.acceleration
        ])