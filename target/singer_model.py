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


class BaseTarget(object):
    def __init__(self, rcs):
        self.rcs = rcs


class SingerModel(BaseTarget):
    def __init__(self, stdAcc, corrAcc, x0, dt, rcs):
        """One dimensional singer model.

        :param stdAcc: Standard deviation of the acceleration noise
        :param corrAcc: Time constant of acceleration correlations (inverse of the actual time)
        :param x0:
        """
        super(SingerModel, self).__init__(rcs)
        self.x0 = x0
        self.x = None
        self.reset()
        self.n = len(self.x)
        self.stdAcc = stdAcc
        self.corrAcc = corrAcc
        self.dt = dt

        self.F = singer_process_matrix(dt, self.corrAcc)
        self.Q = singer_process_covariance(dt, self.corrAcc, self.stdAcc)

    def reset(self):
        self.x = self.x0.reshape(-1, 1)

    def update(self):
        self.x = self.F @ self.x + np.random.multivariate_normal(np.zeros(3), self.Q).reshape(-1, 1)


class SingerModelMD(BaseTarget):
    def __init__(self, stdAcc, corrAcc, x0, dim, dt, rcs):
        """Multi-dimensional Singer model.

        :param stdAcc: Acceleration noise standard deviation
        :param corrAcc: Acceleration noise correlation
        :param x0: Initial state [x, x', x'', y, y', y'', etc.]
        :param dim: How many dimensions used
        """
        super(SingerModelMD, self).__init__(rcs)
        self.models = [SingerModel(stdAcc, corrAcc, x0.flatten()[idx*3:(idx+1)*3], dt, rcs) for idx in range(dim)]
        self.x = None
        self.reset()

        self.n = len(self.x)
        self.stdAcc = stdAcc
        self.corrAcc = corrAcc
        self.dt = dt

        self.F = np.zeros((self.n, self.n))
        self.Q = np.zeros((self.n, self.n))

        for idx, model in enumerate(self.models):
            self.F[idx*3:(idx+1)*3, idx*3:(idx+1)*3] = model.F
            self.Q[idx*3:(idx+1)*3, idx*3:(idx+1)*3] = model.Q

    def reset(self):
        for model in self.models:
            model.reset()
        self.x = np.array([model.x for model in self.models]).reshape(-1, 1)

    def update(self):
        for model in self.models:
            model.update()
        self.x = np.array([model.x for model in self.models]).reshape(-1, 1)
