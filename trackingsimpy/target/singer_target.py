import numpy as np
from trackingsimpy.common.motion_model import singer_process_covariance, singer_process_matrix
from .generic_process import GenericTargetProcess
from .base_target import BaseTarget


class SingerModel(GenericTargetProcess):
    def __init__(self, stdAcc, corrAcc, x0, dt):
        """One dimensional singer model.

        :param stdAcc: Standard deviation of the acceleration noise
        :param corrAcc: Time constant of acceleration correlations (inverse of the actual time)
        :param x0:
        """
        super(SingerModel, self).__init__(x0, F=None, Q=None, order=2, dim=1)

        self.n = len(self.x)
        self.stdAcc = stdAcc
        self.corrAcc = corrAcc
        self.dt = dt

        self.F = singer_process_matrix(dt, self.corrAcc)
        self.Q = singer_process_covariance(dt, self.corrAcc, self.stdAcc)


class SingerModelMD(BaseTarget):
    def __init__(self, stdAcc, corrAcc, x0, dim, dt):
        """Multi-dimensional Singer model.

        :param stdAcc: Acceleration noise standard deviation
        :param corrAcc: Acceleration noise correlation
        :param x0: Initial state [x, x', x'', y, y', y'', etc.]
        :param dim: How many dimensions used
        """
        super(SingerModelMD, self).__init__(order=2, dim=dim)
        self.models = [SingerModel(stdAcc, corrAcc, x0.flatten()[idx*3:(idx+1)*3], dt) for idx in range(dim)]
        self.reset()

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
