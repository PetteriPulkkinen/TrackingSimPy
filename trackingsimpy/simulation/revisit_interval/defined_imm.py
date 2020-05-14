from trackingsimpy.simulation.revisit_interval import BaseRISimulation
from trackingsimpy.tracking import TrackingComputer
from trackingsimpy.radar import PositionRadar
import filterpy.common
from filterpy.kalman import IMMEstimator, KalmanFilter
from trackingsimpy.common.motion_model import constant_turn_rate_matrix, kinematic_state_transition
from trackingsimpy.target import DefinedTargetProcess
from trackingsimpy.common.measurement_model import position_measurement_matrix

import numpy as np


class DefinedIMM(BaseRISimulation):
    ORDER = 1
    DIM = 2
    DT = 0.1
    MAX_STEPS = 700

    def __init__(self, pfa=1e-6, sn0=50, beamwidth=0.02, n_max=20, x0=np.array([10e3, -200.0, 10e3, 0]),
                 P0=np.eye(4) * 1000, theta_accuracy=0.002):
        var = 0.1*9.81**2
        tr1 = -0.08
        tr2 = 0.1
        probs = np.ones(3) / 3
        p_switch = 1.0 / 100.0
        M = np.array([
            [1 - p_switch, p_switch / 2, p_switch / 2],
            [p_switch / 2, 1 - p_switch, p_switch / 2],
            [p_switch / 2, p_switch / 2, 1 - p_switch]
        ])
        filters = list()

        for i in range(3):
            filters.append(filterpy.common.kinematic_kf(self.DIM, self.ORDER, self.DT))
            filters[i].x = x0
            filters[i].Q = filterpy.common.Q_discrete_white_noise(self.DIM, self.DT, var=var, block_size=self.ORDER+1)

        filters[1].F = constant_turn_rate_matrix(tr1, self.DT)
        filters[2].F = constant_turn_rate_matrix(tr2, self.DT)

        st_models = {
            0: kinematic_state_transition(self.DT, self.ORDER, self.DIM),
            200: constant_turn_rate_matrix(tr1, self.DT),
            300: kinematic_state_transition(self.DT, self.ORDER, self.DIM),
            400: constant_turn_rate_matrix(tr2, self.DT),
            500: kinematic_state_transition(self.DT, self.ORDER, self.DIM)
        }

        p_noises = {
            0: filterpy.common.Q_discrete_white_noise(self.DIM, self.DT, var=var, block_size=self.ORDER + 1)
        }
        target = DefinedTargetProcess(x0, st_models, p_noises, self.MAX_STEPS, self.ORDER, self.DIM)
        tracker = IMMEstimator(filters, probs, M)
        radar = PositionRadar(target, sn0, pfa, beamwidth, self.DIM, self.ORDER, angle_accuracy=theta_accuracy)
        computer = TrackingComputer(tracker, radar, n_max, P0=P0)

        super().__init__(computer)


class DefinedCVCAIMM(BaseRISimulation):
    ORDER = 2
    DIM = 2
    DT = 0.1
    MAX_STEPS = 3000

    def __init__(self, pfa=1e-6, sn0=50, beamwidth=0.02, n_max=20, x0=np.array([30e3, -150, 0, 30e3, 150, 0]),
                 theta_accuracy=0.002):

        # Trackers
        probs = np.ones(2) / 2
        p_switch = 1.0 / 1000.0
        M = np.array([
            [1 - p_switch, p_switch],
            [p_switch, 1 - p_switch]])

        F_ca = kinematic_state_transition(self.DT, self.ORDER, self.DIM)
        F_cv = kinematic_state_transition(self.DT, self.ORDER, self.DIM)
        F_cv[:, 2::3] = 0

        g = 9.81
        var = 4 * g ** 2
        Q = filterpy.common.Q_discrete_white_noise(self.DIM, self.DT, var, block_size=self.ORDER + 1)

        kf_cv = KalmanFilter(dim_x=self.DIM * (self.ORDER + 1), dim_z=self.DIM)
        kf_cv.F = F_cv
        kf_cv.Q = Q
        kf_cv.H = position_measurement_matrix(self.DIM, self.ORDER)

        kf_ca = KalmanFilter(dim_x=self.DIM * (self.ORDER + 1), dim_z=self.DIM)
        kf_ca.F = F_ca
        kf_ca.Q = Q
        kf_ca.H = position_measurement_matrix(self.DIM, self.ORDER)

        filters = [kf_cv, kf_ca]
        tracker = IMMEstimator(filters, probs, M)

        # Target
        st_models = {
            0: F_cv,
            1000: F_ca,
            2000: F_cv
        }
        p_noises = {
            0: Q
        }
        target = DefinedTargetProcess(x0, st_models, p_noises, self.MAX_STEPS, self.ORDER, self.DIM)

        # Radar
        radar = PositionRadar(target, sn0, pfa, beamwidth, self.DIM, self.ORDER, angle_accuracy=theta_accuracy)

        # Computer
        P0 = np.zeros((x0.size,) * 2)
        computer = TrackingComputer(tracker, radar, n_max, P0=P0)

        super().__init__(computer)
