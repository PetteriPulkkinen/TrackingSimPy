from .singer_model import BaseTarget
import numpy as np


def preprocess_trajectory_data(trajectory, dim, order):
    traj_trunc = trajectory[:, :dim]  # get the data up to desired dimension
    states = np.zeros((traj_trunc.shape[0], traj_trunc.shape[1] * (order+1)))  # Add additional dims based on the order
    states[:, ::(order+1)] = traj_trunc
    return states


class TargetOnTrajectory(BaseTarget):
    def __init__(self, trajectory):
        self.trajectory = trajectory
        self.length = trajectory.shape[0] - 1
        self.x = trajectory[0].reshape(-1, 1)
        self._idx = 0

    def update(self):
        self._idx += 1
        if self._idx >= len(self.trajectory):
            raise RuntimeError("Target trajectory index is over the the trajectory length!")
        self.x = self.trajectory[self._idx].reshape(-1, 1)

    def reset(self):
        self._idx = 0
        self.x = self.trajectory[0].reshape(-1, 1)
