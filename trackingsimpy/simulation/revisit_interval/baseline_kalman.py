from trackingsimpy.simulation.revisit_interval.base_ri_sim import BaseRISimulation
from trackingsimpy.simulation.revisit_interval.baseline import Baseline
from trackingsimpy.trajectories import get_file_list, load_trajectory
import filterpy.common
from trackingsimpy.radar import PositionRadar
from trackingsimpy.tracking import TrackingComputer
from trackingsimpy.target import TrajectoryTarget
import filterpy.common
import numpy as np


class BaselineKalman(Baseline):
    ORDER = 1
    DIM = 2
    DT = 0.01

    def __init__(self, n_max=20, var=10, traj_idx=0, P0=None, beamwidth=0.02, pfa=1e-6, sn0=50, theta_accuracy=0.002):
        tracker = filterpy.common.kinematic_kf(self.DIM, self.ORDER, self.DT)
        tracker.Q = filterpy.common.Q_discrete_white_noise(self.DIM, self.DT, var=var, block_size=self.ORDER+1)
        super().__init__(
            tracker=tracker,
            n_max=n_max,
            traj_idx=traj_idx,
            P0=P0,
            beamwidth=beamwidth,
            pfa=pfa,
            sn0=sn0,
            theta_accuracy=theta_accuracy
        )


class BenchmarkWithKalmanFilter(object):
    DT = 0.01

    def __init__(self):
        dt = 0.01
        dim = 2
        order = 1
        var = (4.5 * 9.81) ** 2
        self.sims = list()

        for i in range(6):
            trajectory = load_trajectory(get_file_list()[i], order, dim=dim)
            target = TrajectoryTarget(trajectory, order, dim)
            tracker = filterpy.common.kinematic_kf(dim, order, dt=dt)
            tracker.Q = filterpy.common.Q_discrete_white_noise(dim, dt, var=var, block_size=order + 1)
            radar = PositionRadar(target, 50, 1e-6, 0.02, dim, order, angle_accuracy=0.002)
            computer = TrackingComputer(tracker, radar, 20, x0=target.x, P0=np.eye(4) * 0)
            sim = BaseRISimulation(computer)
            sim.DT = dt
            self.sims.append(sim)

