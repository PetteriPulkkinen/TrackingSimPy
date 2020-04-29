from trackingsimpy.tracking import TrackingComputer
from trackingsimpy.simulation import Saver
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class BaseRISimulation(object):
    def __init__(self, computer: TrackingComputer, saver_ds=None):
        self.target = computer.radar.target
        self.radar = computer.radar
        self.tracker = computer.tracker
        self.computer = computer
        self._save = False

        saver_ds = {
            self.target: "x",
            self.radar: "angle_error",
            self.tracker: "x_prior",
            computer: ["y", "current_time", "z", "yn", "snr"]
        }

        if saver_ds is not None:
            self.saver = Saver(saver_ds)
        else:
            self.saver = None

    def reset(self):
        self.target.reset()
        self.computer.initialize(x0=self.target.x)
        if self._save:
            self.saver.reset()

    def step(self, revisit_interval):
        for _ in range(revisit_interval):
            self.computer.predict()
            trajectory_ends = self.target.trajectory_ends()
            if not trajectory_ends:
                self.target.update()
            else:
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

    def get_animation(self, interval_ms=1):
        self.saver.convert_to_numpy()

        fig, ax = plt.subplots(1, 1)

        pos = self.saver[self.target]['x'][:, [0, 2]]
        pos_est = self.saver[self.tracker]['x_prior'][:, [0, 2]]
        meas = self.saver[self.computer]['z']

        ax.plot(pos[:, 0], pos[:, 1], '--', label='trajectory')
        pos_line = ax.plot([], [], 'o', label='position')[0]
        pos_est_line = ax.plot([], [],  'o', label='predicted position')[0]
        meas_line = ax.plot([], [], 'x', label='measurement', alpha=0.2)[0]
        plt.legend()
        plt.grid(True)

        def animate(idx):
            pos_line.set_data(pos[idx, 0], pos[idx, 1])
            pos_est_line.set_data(pos_est[idx, 0], pos_est[idx, 1])
            meas_line.set_data(meas[:idx, 0], meas[:idx, 1])
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
        self.saver.convert_to_numpy()

        fig, ax = plt.subplots(1, 1)

        pos = self.saver[self.target]['x'][:, [0, 2]]
        pos_est = self.saver[self.tracker]['x_prior'][:, [0, 2]]
        meas = self.saver[self.computer]['z']

        ax.plot(meas[:, 0], meas[:, 1], 'x', label='measurement', alpha=0.2)
        ax.plot(pos[:, 0], pos[:, 1], '-', label='position')
        ax.plot(pos_est[:, 0], pos_est[:, 1], '-', label='predicted position')
        
        plt.legend()
        plt.grid(True)
        plt.show()
