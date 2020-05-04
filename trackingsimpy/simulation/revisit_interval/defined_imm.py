from trackingsimpy.simulation.revisit_interval import BaseRISimulation
from trackingsimpy.tracking import TrackingComputer
from trackingsimpy.radar import PositionRadar
import filterpy.common
from filterpy.kalman import IMMEstimator
from trackingsimpy.common.motion_model import constant_turn_rate_matrix, kinematic_state_transition
from trackingsimpy.target import DefinedTargetProcess

import numpy as np


class DefinedIMM(BaseRISimulation):
    ORDER = 1
    DIM = 2
    DT = 0.01
    MAX_STEPS = 4000

    def __init__(self, pfa=1e-6, sn0=50, beamwidth=0.02, n_max=20, x0=np.array([10e3, -200.0, 10e3, 0]),
                 P0=np.eye(4) * 1000):

        probs = np.ones(3) / 3
        p_switch = 1.0 / 500.0
        M = np.array([
            [1 - p_switch, p_switch / 2, p_switch / 2],
            [p_switch / 2, 1 - p_switch, p_switch / 2],
            [p_switch / 2, p_switch / 2, 1 - p_switch]
        ])
        filters = list()

        for i in range(3):
            filters.append(filterpy.common.kinematic_kf(self.DIM, self.ORDER, self.DT))
            filters[i].x = x0
            filters[i].Q = filterpy.common.Q_discrete_white_noise(self.DIM, self.DT, var=1000, block_size=self.ORDER+1)

        filters[1].F = constant_turn_rate_matrix(-0.3, self.DT)
        filters[2].F = constant_turn_rate_matrix(0.8, self.DT)

        st_models = {
            0: kinematic_state_transition(self.DT, self.ORDER, self.DIM),
            1000: constant_turn_rate_matrix(-0.3, self.DT),
            1500: kinematic_state_transition(self.DT, self.ORDER, self.DIM),
            2500: constant_turn_rate_matrix(0.8, self.DT),
            3000: kinematic_state_transition(self.DT, self.ORDER, self.DIM)
        }

        p_noises = {
            0: filterpy.common.Q_discrete_white_noise(self.DIM, self.DT, var=1000, block_size=self.ORDER + 1)
        }
        target = DefinedTargetProcess(x0, st_models, p_noises, self.MAX_STEPS, self.ORDER, self.DIM)
        tracker = IMMEstimator(filters, probs, M)
        radar = PositionRadar(target, sn0, pfa, beamwidth, self.DIM, self.ORDER)
        computer = TrackingComputer(tracker, radar, n_max, P0=P0)

        super().__init__(computer)
