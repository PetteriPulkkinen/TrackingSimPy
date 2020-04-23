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

    def __init__(self, tracker, k_min, k_max, n_max, traj_idx, P0, beamwidth, pfa, sn0):
        if P0 is None:
            P0 = np.eye((self.ORDER + 1) * self.DIM) * 1000

        target = TrajectoryTarget(
            load_trajectory(get_file_list()[traj_idx], self.ORDER, dim=self.DIM), self.ORDER, self.DIM)
        radar = PositionRadar(target, sn0, pfa, beamwidth, self.DIM, self.ORDER)
        tracker = tracker
        computer = TrackingComputer(tracker, radar, n_max=n_max, P0=P0)

        saver_ds = {
            target: "x",
            radar: "angle_error",
            tracker: "x",
            computer: ["y", "current_time", "z", "yn", "snr"]
        }

        super().__init__(computer=computer, k_min=k_min, k_max=k_max, saver_ds=saver_ds)
