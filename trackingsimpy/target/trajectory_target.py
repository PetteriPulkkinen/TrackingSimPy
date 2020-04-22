from .base_target import BaseTarget


class TargetOnTrajectory(BaseTarget):
    def __init__(self, trajectory, order, dim):
        super(TargetOnTrajectory, self).__init__(order=order, dim=dim)
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
