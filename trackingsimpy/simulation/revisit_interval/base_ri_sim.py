from trackingsimpy.tracking import TrackingComputer
from trackingsimpy.common import Saver
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from filterpy.kalman import IMMEstimator
from trackingsimpy.common.trigonometrics import pos_to_angle_error_2D

import numpy as np


class BaseRISimulation(object):
    G = 9.81

    def __init__(self, computer: TrackingComputer):
        self.target = computer.radar.target
        self.radar = computer.radar
        self.tracker = computer.tracker
        self.computer = computer
        self._save = False
        self.order = self.target.order
        self.dim = self.target.dim

        self.revisit_interval = 1

        saver_ds_sts = {
            self: ["revisit_interval"],
            self.target: ["x"],
            self.tracker: ["x_prior", "x_post", "P_prior", "P_post"],
            self.radar: ["angle_error", "angle_std"],
            computer: ["y", "theta", "theta_smoothed", "current_time", "z", "yn", "snr", "n_missed"]
        }

        saver_ds_fts = {
            self.target: ["x"],
            self.tracker: ["x_prior", "P_prior"],
        }

        if type(self.tracker) is IMMEstimator:
            saver_ds_sts[self.tracker].append("mu")

        self.saver = Saver(saver_ds_sts)
        self.saver_fts = Saver(saver_ds_fts)

    def reset(self):
        self.target.reset()
        self.computer.initialize(x0=self.target.x)
        if self._save:
            self.saver.reset()
            self.saver_fts.reset()

    def step(self, revisit_interval):
        self.revisit_interval = revisit_interval
        for _ in range(revisit_interval):
            self.computer.predict()
            self.target.update()
            if self._save:
                self.saver_fts.save()

            trajectory_ends = self.target.trajectory_ends()
            if trajectory_ends:
                break

        update_successful, n_missed = self.computer.update_track()

        if self._save:
            self.saver.save()

        return update_successful, n_missed, trajectory_ends

    def enable_saving(self):
        if self.saver is None:
            raise RuntimeError("Saving structure needs to be defined if saving is enabled!")
        self._save = True

    def disable_saving(self):
        self._save = False

    def get_animation(self, interval_ms=1, meas_alpha=0.2):
        assert self.dim == 2
        self.saver.convert_to_numpy()

        fig, ax = plt.subplots(1, 1)

        pos = self.saver[self.target]['x'][:, ::self.order+1]
        pos_est = self.saver[self.tracker]['x_prior'][:, ::self.order+1]
        meas = self.saver[self.computer]['z']

        ax.plot(pos[:, 0], pos[:, 1], '--', label='trajectory')
        pos_line = ax.plot([], [], 'o', label='position')[0]
        pos_est_line = ax.plot([], [],  'o', label='predicted position')[0]
        meas_line = ax.plot([], [], 'x', label='measurement', alpha=meas_alpha)[0]
        plt.legend()
        plt.grid(True)

        def animate(idx):
            pos_line.set_data(pos[idx, 0], pos[idx, 1])
            pos_est_line.set_data(pos_est[idx, 0], pos_est[idx, 1])
            meas_line.set_data(meas[:idx+1, 0], meas[:idx+1, 1])
            return (pos_line, pos_est_line, meas_line)

        def init():
            pos_line.set_data([], [])
            pos_est_line.set_data([], [])
            meas_line.set_data([], [])
            return (pos_line, pos_est_line, meas_line)

        # call the animator. blit=True means only re-draw the parts that have changed.
        n_frames = meas.shape[0]
        anim = FuncAnimation(fig, animate, init_func=init,
                             frames=n_frames, interval=interval_ms, blit=True)
        return anim

    def visualize(self):
        assert self.dim == 2
        self.saver.convert_to_numpy()

        fig, ax = plt.subplots(1, 1)

        pos = self.saver[self.target]['x'][:, ::self.order + 1]
        pos_est = self.saver[self.tracker]['x_prior'][:, ::self.order + 1]
        pos_corr = self.saver[self.tracker]['x_post'][:, ::self.order + 1]
        meas = self.saver[self.computer]['z']

        ax.plot(meas[:, 0], meas[:, 1], 'x', label='measurement', alpha=0.2)
        ax.plot(pos[:, 0], pos[:, 1], '-', label='position')
        ax.plot(pos_est[:, 0], pos_est[:, 1], '-', label='prior position')
        ax.plot(pos_corr[:, 0], pos_corr[:, 1], '-', label='post position')
        
        plt.legend()
        plt.grid(True)
        plt.show()

    def pos_pred_error(self):
        self.saver_fts.convert_to_numpy()

        fig, ax = plt.subplots(1, 1)
        pos = self.saver_fts[self.target]['x'][:, ::self.order + 1]
        pos_pred = self.saver_fts[self.tracker]['x_prior'][:, ::self.order + 1]

        plt.plot(np.linalg.norm(pos - pos_pred, axis=1))
        plt.xlabel('Step')
        plt.ylabel('Error [m]')
        plt.grid(True)
        plt.show()

    def angle_pred_error(self):
        self.saver_fts.convert_to_numpy()

        fig, ax = plt.subplots(1, 1)
        pos = self.saver_fts[self.target]['x'][:, ::self.order + 1]
        pos_pred = self.saver_fts[self.tracker]['x_prior'][:, ::self.order + 1]

        error = np.array([pos_to_angle_error_2D(pos1, pos2) for pos1, pos2 in zip(pos, pos_pred)])
        plt.plot(error/self.radar.beamwidth)
        plt.xlabel('Step')
        plt.ylabel('Relative error')
        plt.title('Angle prediction error relative to beamwidth')
        plt.grid(True)
        plt.show()

    def tracking_load(self):
        self.saver.convert_to_numpy()

        fig, ax = plt.subplots(1, 1)
        n_missed = self.saver[self.computer]['n_missed']
        ri = self.saver[self]['revisit_interval']
        upd_time = self.saver[self.computer]['current_time']

        load = (n_missed+1) / ri

        load_rep = np.repeat(load, 2)
        upd_time_rep = np.zeros(upd_time.size*2)
        upd_time_rep[1:] = np.repeat(upd_time, 2)[:-1]

        plt.plot(upd_time_rep, load_rep)
        plt.xlabel('Step')
        plt.ylabel('Load')
        plt.grid(True)
        plt.show()

    def revisit_interval_schedule(self):
        self.saver.convert_to_numpy()

        fig, ax = plt.subplots(1, 1)

        ri = self.saver[self]['revisit_interval']
        upd_time = self.saver[self.computer]['current_time']

        ri_rep = np.repeat(ri, 2)
        upd_time_rep = np.zeros(upd_time.size * 2)
        upd_time_rep[1:] = np.repeat(upd_time, 2)[:-1]

        plt.plot(upd_time_rep*self.DT, ri_rep*self.DT)
        plt.xlabel('Time [s]')
        plt.ylabel('Revisit interval [s]')
        plt.grid(True)
        plt.show()
