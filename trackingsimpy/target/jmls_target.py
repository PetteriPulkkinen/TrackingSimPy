import numpy as np
from trackingsimpy.common import MarkovChain


class JMLSTarget(object):
    def __init__(self, x0, T, S, noise):
        """
        Creates two dimensional target model with Markovian control signal
        :param x0: Initial kinematic state
        :param T: Transition probabilities for the Markov chain
        :param S: Control states (NxM) where N is number of states and M is number of features
        """
        self.x0 = x0
        self.F = lambda dt: np.array([
                    [1, 0, dt, 0],
                    [0, 1, 0, dt],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
        ])
        self.B = lambda dt: np.array([
                    [dt**2/2, 0],
                    [0, dt**2/2],
                    [dt, 0],
                    [0, dt],
        ])
        self.x = x0
        self.Q = lambda dt: noise**2 * np.array([
                    [dt**2, 0, 0, 0],
                    [0, dt**2, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0]
        ])

        self.mc = MarkovChain(T=T, S=S, x0=0)
        
    def reset(self, x0=None):
        if x0 is None:
            self.x = self.x0
        else:
            self.x = x0

    def update(self, dt):
        ctrl = self.mc.observe()
        self.x = \
            self.F(dt) @ self.x + self.B(dt) @ ctrl + np.random.multivariate_normal(np.zeros_like(self.x), self.Q(dt))

    def get_state(self):
        return self.x

    def get_position(self):
        return self.x[:2]
