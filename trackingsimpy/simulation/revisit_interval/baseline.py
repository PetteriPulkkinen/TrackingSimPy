from trackingsimpy.simulation.revisit_interval import BaseRISimulation
from trackingsimpy.radar import PositionRadar
from trackingsimpy.target import TrajectoryTarget
from trackingsimpy.tracking import TrackingComputer
from trackingsimpy.trajectories import get_file_list, load_trajectory

import numpy as np


class Baseline(BaseRISimulation):
    ORDER = 1
    DIM = 2
    DT = 0.01

    def __init__(self, tracker, n_max, traj_idx, P0, beamwidth, pfa, sn0, theta_accuracy):
        if P0 is None:
            P0 = np.eye((self.ORDER + 1) * self.DIM) * 1000

        target = TrajectoryTarget(
            load_trajectory(get_file_list()[traj_idx], self.ORDER, dim=self.DIM), self.ORDER, self.DIM)
        radar = PositionRadar(target, sn0, pfa, beamwidth, self.DIM, self.ORDER, angle_accuracy=theta_accuracy)
        computer = TrackingComputer(tracker, radar, n_max=n_max, P0=P0)

        super().__init__(computer=computer)


