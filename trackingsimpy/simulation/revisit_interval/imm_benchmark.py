import numpy as np
from trackingsimpy.simulation.revisit_interval.baseline import BaseRISimulation
from trackingsimpy.common.motion_model import kinematic_state_transition
from trackingsimpy.target import load_benchmark_target
from trackingsimpy.radar import PositionRadar
from trackingsimpy.tracking import TrackingComputer
from scipy.linalg import block_diag
from filterpy.kalman import KalmanFilter, IMMEstimator
from trackingsimpy.common.measurement_model import position_measurement_matrix


class IMMBenchmark(BaseRISimulation):
    ORDER = 2
    DIM = 2
    DT = 0.1

    def __init__(self, var_cv=737.5, var_ca=73.7, p_switch=0.028, traj_idx=0):
        F_ca = kinematic_state_transition(self.DT, self.ORDER, self.DIM)
        F_cv = F_ca.copy()
        F_cv[:, 2::3] = 0
    
        G = np.array([[1/2*self.DT**2, self.DT, 1]]).T
        A = G @ G.T

        Q_ca = block_diag(A, A) * var_ca
        Q_cv = block_diag(A, A) * var_cv
        Q_cv[:, 2::3] = 0
        Q_cv[2::3, :] = 0

        kf_ca = KalmanFilter(dim_x=self.DIM * (self.ORDER + 1), dim_z=self.DIM)
        kf_ca.F = F_ca
        kf_ca.Q = Q_ca
        kf_ca.H = position_measurement_matrix(self.DIM, self.ORDER)

        kf_cv = KalmanFilter(dim_x=self.DIM * (self.ORDER + 1), dim_z=self.DIM)
        kf_cv.F = F_cv
        kf_cv.Q = Q_cv
        kf_cv.H = position_measurement_matrix(self.DIM, self.ORDER)

        filters = [kf_cv, kf_ca]
        mu = [0.5, 0.5]
        M = np.array([
            [1 - p_switch, p_switch],
            [p_switch, 1 - p_switch]
        ])

        tracker = IMMEstimator(filters, mu, M)

        target = load_benchmark_target(traj_idx, self.ORDER, self.DIM, skip_k=10)

        sn0 = 50.0
        pfa = 1e-6
        beamwidth = 0.02

        radar = PositionRadar(target, sn0, pfa, beamwidth, self.DIM, self.ORDER)

        n_max = 20
        P0 = np.zeros((6, 6), dtype=float)
        target.reset()
        x0 = target.x

        computer = TrackingComputer(tracker, radar, n_max, x0=x0, P0=P0)

        super().__init__(computer=computer)
