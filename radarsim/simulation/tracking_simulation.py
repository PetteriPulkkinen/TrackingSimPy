from radarsim.tracking import TrackingComputer
from radarsim.radar import TrackingRadar
from radarsim.target import TargetOnTrajectory
from radarsim.trajectories import get_file_list, load_trajectory
from filterpy.common import kinematic_kf, Q_discrete_white_noise, Saver

import numpy as np


class TrackingSimulation(object):
    DIM = 2
    ORDER = 1
    FS = 100

    def __init__(self, n_max=20, x0=None, P0=None, traj_idx=0, sn0=50.0, pfa=1e-4,
                 beamwidth=0.01, variance=1.0, save=False, k_max=1000, k_min=1, p_loss=500, alpha=0.98):
        trajectory = load_trajectory(get_file_list()[traj_idx], self.ORDER, dim=self.DIM)
        self.target = TargetOnTrajectory(trajectory)

        tracker = kinematic_kf(self.DIM, self.ORDER, dt=1/self.FS)
        tracker.Q = Q_discrete_white_noise(
            self.DIM, 1/self.FS, var=variance, block_size=self.ORDER+1)

        self.radar = TrackingRadar(self.target, sn0, pfa, beamwidth, self.DIM, self.ORDER)

        self.tracker_computer = TrackingComputer(tracker, self.radar, n_max=n_max, alpha=alpha)
        self.x0 = x0
        self.P0 = P0

        self.k_max = k_max
        self.k_min = k_min
        self.p_loss = p_loss  # Penalty for lost target

        self.save = save
        self.tracking_saver = None
        self.computer_saver = None
        self.radar_saver = None

    def reset(self):
        self.target.reset()
        if self.x0 is None:
            x0 = self.target.x
        if self.P0 is None:
            P0 = np.eye(self.target.x.size)*100

        obs = np.ones(shape=self.DIM, dtype=np.float) * 1.2
        self.tracker_computer.initialize(x0=x0, P0=P0, yn0_smoothed=obs)
        if self.save:
            self.tracking_saver = Saver(self.tracker_computer.tracker, skip_private=True)
            self.computer_saver = Saver(
                self.tracker_computer,
                ignore=(
                    'tracker',
                    'radar',
                    'n_max',
                    'alpha'
                ),
                skip_private=True
            )
            self.radar_saver = Saver(
                self.radar,
                ignore=(
                    'sn0',
                    'pfa',
                    'n_max',
                    'target',
                    'R',
                    'H',
                    'beamwidth'
                ),
                skip_private=True
            )
        return obs

    def step(self, revisit_interval):
        for _ in range(revisit_interval):
            self.tracker_computer.predict()
            self.target.update()
            if self.save:
                self.tracking_saver.save()

        update_successful, n_missed = self.tracker_computer.update_track()

        if self.save:
            self.computer_saver.save()
            self.radar_saver.save()

        reward = self._reward(update_successful, n_missed, revisit_interval)
        obs = self._observation(update_successful)

        # Ensure that the trajectory does not overflow at next step
        trajectory_ends = (self.target.current_idx + self.k_max) >= len(self.target.trajectory)

        if trajectory_ends or not update_successful:
            done = True
        else:
            done = False

        return obs, reward, done, {}

    def _observation(self, update_successful):
        if update_successful:
            return self.tracker_computer.yn
        else:
            return np.ones(shape=self.DIM, dtype=np.float) * 100

    def _reward(self, update_successful, n_missed, revisit_interval):
        if update_successful:
            return - (n_missed + 1) / revisit_interval
        else:
            return - self.p_loss
