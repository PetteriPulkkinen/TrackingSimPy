import numpy as np
from trackingsimpy.tracking.computer import TrackingComputer
from trackingsimpy.common.trigonometrics import angular_uncertainty_2D
import copy


class BaseUpdatePolicy(object):
    def __init__(self):
        pass

    def get_revisit_interval(self):
        raise NotImplementedError


class RandomUpdatePolicy(BaseUpdatePolicy):
    def __init__(self, intervals):
        super().__init__()
        self.intervals = intervals

    def get_revisit_interval(self):
        return np.random.choice(self.intervals)


class ConstantUpdatePolicy(BaseUpdatePolicy):
    def __init__(self, interval):
        super().__init__()
        self.interval = interval

    def get_revisit_interval(self):
        return self.interval


class CovarianceBasedPolicy(BaseUpdatePolicy):
    def __init__(self, computer: TrackingComputer, v0, ri_min, ri_max):
        super().__init__()
        self.computer = computer
        self.ri_min = ri_min
        self.ri_max = ri_max
        self.v0 = v0  # threshold angular uncertainty
        self.order = computer.radar.target.order

    def get_revisit_interval(self):
        tracker = copy.deepcopy(self.computer.tracker)
        ri = 0
        for _ in range(self.ri_max):
            tracker.predict()
            ri += 1
            angle_std = angular_uncertainty_2D(tracker.x, tracker.P, self.order)
            if angle_std > self.v0 * self.computer.radar.beamwidth:
                ri -= 1
                break

        return np.max([self.ri_min, ri])
