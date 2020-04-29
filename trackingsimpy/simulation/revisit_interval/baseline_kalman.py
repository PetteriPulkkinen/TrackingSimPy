from trackingsimpy.simulation.revisit_interval.baseline import Baseline
import filterpy.common


class BaselineKalman(Baseline):
    ORDER = 1
    DIM = 2
    DT = 0.01

    def __init__(self, n_max=20, var=10, traj_idx=0, P0=None, beamwidth=0.02, pfa=1e-6, sn0=50):
        tracker = filterpy.common.kinematic_kf(self.DIM, self.ORDER, self.DT)
        tracker.Q = filterpy.common.Q_discrete_white_noise(self.DIM, self.DT, var=var, block_size=self.ORDER+1)
        super().__init__(
            tracker=tracker,
            n_max=n_max,
            traj_idx=traj_idx,
            P0=P0,
            beamwidth=beamwidth,
            pfa=pfa,
            sn0=sn0
        )
