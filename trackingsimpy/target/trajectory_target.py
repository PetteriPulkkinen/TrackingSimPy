from .singer_target import BaseTarget
import numpy as np


def preprocess_trajectory_data(trajectory, dim, order):
    traj_trunc = trajectory[:, :dim]  # get the data up to desired dimension
    # Add additional dims based on the order
    states = np.zeros((traj_trunc.shape[0], traj_trunc.shape[1] * (order+1)))
    states[:, ::(order+1)] = traj_trunc
    return states


class TargetOnTrajectory(BaseTarget):
    def __init__(self, trajectory):
        self.trajectory = trajectory
        self.length = trajectory.shape[0] - 1
        self.x = trajectory[0].reshape(-1, 1)
        self.current_idx = 0

    def update(self):
        self.current_idx += 1
        if self.current_idx >= len(self.trajectory):
            raise RuntimeError("Target trajectory index is over the the trajectory length!")
        self.x = self.trajectory[self.current_idx].reshape(-1, 1)

    def reset(self):
        self.current_idx = 0
        self.x = self.trajectory[0].reshape(-1, 1)
