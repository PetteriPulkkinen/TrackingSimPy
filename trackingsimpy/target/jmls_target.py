import numpy as np
from trackingsimpy.common import MarkovChain
from .generic_process import GenericTargetProcess
from trackingsimpy.common.motion_model import kinematic_state_transition, acceleration_control_matrix
import filterpy.common


class JMLSTarget(GenericTargetProcess):
    def __init__(self, dt, x0, T, S, noise, dim):
        """
        Creates two dimensional target model with Markovian control signal
        :param x0: Initial kinematic state
        :param T: Transition probabilities for the Markov chain
        :param S: Control states (NxM) where N is number of states and M is number of features
        """

        super().__init__(x0=x0, F=None, Q=None, order=1, dim=dim)

        self.F = kinematic_state_transition(dt, self.order, self.dim)
        self.B = acceleration_control_matrix(dt, self.dim)
        self.Q = filterpy.common.Q_discrete_white_noise(self.order + 1, dt=dt, var=noise, block_size=dim)

        self.mc = MarkovChain(T=T, S=S, x0=0)

    def update(self):
        super().update()
        ctrl = self.mc.observe()
        self.x += self.B @ ctrl
