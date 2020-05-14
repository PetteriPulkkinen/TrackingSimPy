from trackingsimpy.simulation.revisit_interval import BaseRISimulation
from trackingsimpy.tracking import TrackingComputer
from trackingsimpy.radar import PositionRadar
import filterpy.common
from trackingsimpy.common.motion_model import kinematic_state_transition
from trackingsimpy.target import DefinedTargetProcess

import numpy as np


class DefinedKalman(BaseRISimulation):
    ORDER = 1
    DIM = 2
    DT = 0.1
    MAX_STEPS = 2_000

    def __init__(self, pfa=1e-6, sn0=50, beamwidth=0.02, n_max=20, var=30**2, x0=np.array([30e3, -100.0, 30e3, -100.0]),
                 P0=np.zeros((4, 4)), theta_accuracy=0.02):

        st_models = {
            0: kinematic_state_transition(self.DT, self.ORDER, self.DIM),
        }
        p_noises = {
            0: filterpy.common.Q_discrete_white_noise(self.DIM, self.DT, var=var, block_size=self.ORDER + 1)
        }
        target = DefinedTargetProcess(x0, st_models, p_noises, self.MAX_STEPS, self.ORDER, self.DIM)
        tracker = filterpy.common.kinematic_kf(dim=self.DIM, order=self.ORDER, dt=self.DT)
        tracker.Q = filterpy.common.Q_discrete_white_noise(self.DIM, self.DT, var, self.ORDER+1)
        radar = PositionRadar(target, sn0, pfa, beamwidth, self.DIM, self.ORDER, angle_accuracy=theta_accuracy)
        computer = TrackingComputer(tracker, radar, n_max, x0=x0, P0=P0)

        super().__init__(computer)
