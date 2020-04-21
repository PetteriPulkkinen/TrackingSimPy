import numpy as np
from trackingsimpy.common.motion_model import singer_process_covariance, singer_process_matrix


class BaseTarget(object):
    def reset(self):
        raise NotImplementedError

    def update(self):
        raise NotImplementedError


class SingerModel(BaseTarget):
    def __init__(self, stdAcc, corrAcc, x0, dt):
        """One dimensional singer model.

        :param stdAcc: Standard deviation of the acceleration noise
        :param corrAcc: Time constant of acceleration correlations (inverse of the actual time)
        :param x0:
        """
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
    def __init__(self, stdAcc, corrAcc, x0, dim, dt):
        """Multi-dimensional Singer model.

        :param stdAcc: Acceleration noise standard deviation
        :param corrAcc: Acceleration noise correlation
        :param x0: Initial state [x, x', x'', y, y', y'', etc.]
        :param dim: How many dimensions used
        """
        self.models = [SingerModel(stdAcc, corrAcc, x0.flatten()[idx*3:(idx+1)*3], dt) for idx in range(dim)]
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
