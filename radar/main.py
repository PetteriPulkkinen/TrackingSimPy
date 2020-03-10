import numpy as np
import uuid
from collections import namedtuple

RadarReturn = namedtuple('radar_return', ['detection', 'signal_quality', 'state'])
RadarSignal = namedtuple('radar_signal', ['power'])


class SimulationObject(object):
    def __init__(self):
        self.ID = uuid.uuid1()


class Platform(SimulationObject):
    def __init__(self, dim):
        self.dim = self.dim
        self.position = np.zeros(dim)
        self.velocity = np.zeros(dim)
        self.acceleration = np.zeros(dim)

    def state(self):
        return np.array([
            self.platform.position,
            self.platform.velocity,
            self.platform.acceleration
        ])


class Environment(SimulationObject):
    def __init__(self, targets=None):
        self.targets = targets
        self.thermal_noise_power = 0

    def add_target(self, target):
        self.targets[target.ID] = target

    def propagate(self, signal, from_platform):
        radar_returns = list()
        for id, target in self.targets.items():
            signal.power
            radar_returns.append(RadarReturn(False, np.zeros(from_platform.dim)))
        return radar_returns

    @staticmethod
    def path_loss(from_platform, to_platform, via_platform=None):
        if via_platform is None:
            distance = np.linalg.norm(from_platform.position - to_platform.position)
        else:
            # Bi-static case
            raise NotImplementedError
        return 1/(distance**4)


class Radar(Platform):
    def __init__(self, environment, dim=2):
        super(Radar, self).__init__(dim)
        self.environment = environment
        self.

    def illuminate(self, angle):
        radar_returns = self.environment.propagate(self)
        return RadarReturn(False, np.zeros((1, 1)))


class Target(Platform):
    def __init__(self, dim=2):
        super(Target, self).__init__(dim)

    @staticmethod
    def rcs():
        return 1