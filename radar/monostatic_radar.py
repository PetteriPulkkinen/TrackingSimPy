import numpy as np


class MonostaticRadar(object):
    def __init__(self):
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        self.R = lambda dt: np.array([
            [dt, 0],
            [0, dt]
        ])

    def measure(self, target, dt):
        """
        :param target: Target state to be measured
        :param dt: Time difference from last measurement
        :return:
        """
        y = self.H @ target.get_state()
        return y + np.random.multivariate_normal(np.zeros_like(y), self.R(dt))
