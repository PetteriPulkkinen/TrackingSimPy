from .singer_model import BaseTarget


class TargetOnTrajectory(BaseTarget):
    def __init__(self, trajectory):
        self.trajectory = trajectory
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
