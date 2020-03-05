import numpy as np


def singer_process_matrix(dt, corr_acc):
    return np.array([
            [1, dt, (corr_acc*dt - 1 + np.exp(-corr_acc*dt) / np.power(corr_acc, 2))],
            [0, 1 , (1 - np.exp(-corr_acc*dt)/corr_acc)],
            [0, 0 , np.exp(-corr_acc*dt)],
        ])


def singer_process_covariance(dt, corr_acc, std_acc):
    return 2*corr_acc*std_acc* np.array([
            [dt ** 5 / 20, dt ** 4 / 8, dt ** 3 / 6],
            [dt ** 4 / 8 , dt ** 3 / 3, dt ** 2 / 2],
            [dt ** 3 / 6 , dt ** 2 / 2, dt         ]
        ])


class SingerModel(object):
    def __init__(self, stdAcc, corrAcc, x0):
        """One dimensional singer model.

        :param stdAcc: Standard deviation of the acceleration noise
        :param corrAcc: Time constant of acceleration correlations (inverse of the actual time)
        :param x0:
        """
        self.x0 = x0
        self.x = x0
        self.n = len(self.x)
        self.stdAcc = stdAcc
        self.corrAcc = corrAcc

        self.F = lambda dt: singer_process_matrix(dt, self.corrAcc)
        self.Q = lambda dt: singer_process_covariance(dt, self.corrAcc, self.stdAcc)

    def reset(self):
        self.x = self.x0

    def update(self, dt):
        self.x = self.F(dt)@self.x + np.random.multivariate_normal(np.zeros(3), self.Q(dt))
