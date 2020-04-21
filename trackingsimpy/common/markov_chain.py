import numpy as np


class MarkovChain(object):
    def __init__(self, T, S, x0=None):
        """
        :param T: Transition probabilities (N, N) where N is number of states
        :param S: States with shape (N, *) where N is number of states and * additional dimensions
        :param x0: Initial state as integer (default None:=random)
        """
        self.T = T
        self.S = S
        self.n = self.T.shape[0]

        if x0 is None:
            self.x = np.random.randint(self.n)
        else:
            self.x = x0

    def observe(self):
        """
        :return: state presentation
        """
        obs = self.S[self.x]
        self.x = np.random.choice(np.arange(self.n), p=self.T[self.x])

        return obs
