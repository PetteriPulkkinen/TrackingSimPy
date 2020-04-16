from tensorforce import Environment
from radarsim.simulation import TrackingSimulation
import numpy as np


class CustomEnvironment(Environment):

    def __init__(self):
        self.sim = TrackingSimulation(
            n_max=20,
            traj_idx=2,
            variance=50,
            k_min=1,
            k_max=100,
            p_loss=5000,
            save=False,
            beamwidth=0.025
        )
        self.n_actions = 9
        self.delta_lim = 100
        self.dim_observations = 2
        self.revisit_interval = 1
        super().__init__()

    def states(self):
        return dict(type='float', shape=(self.dim_observations,))

    def actions(self):
        return dict(type='int', num_values=self.n_actions)

    def reset(self):
        return self.sim.reset()

    def execute(self, actions):
        delta = int(self.delta_lim * (2*actions - self.n_actions + 1) / (self.n_actions - 1))
        self.revisit_interval += delta
        self.revisit_interval = np.max([self.sim.k_min, self.revisit_interval])
        self.revisit_interval = np.min([self.sim.k_max, self.revisit_interval])
        obs, reward, done, _ = self.sim.step(self.revisit_interval)
        return obs, done, reward
