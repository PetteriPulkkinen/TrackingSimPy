import numpy as np


class BaseUpdatePolicy(object):
    def choose(self):
        raise NotImplementedError


class RandomUpdatePolicy(BaseUpdatePolicy):
    def __init__(self, intervals):
        self.intervals = intervals

    def choose(self):
        return np.random.choice(self.intervals)
    
    
class ConstantUpdatePolicy(BaseUpdatePolicy):
    def __init__(self, interval):
        self.interval = interval
        
    def choose(self):
        return self.interval


class ResidualUpdatePolicy(object):
    def __init__(self):
        self.K = 1

    def update(self, x, y, r):
        self.K = np.max([1, np.round(self.K*np.power(r/np.abs(x - y), 1/2))])


if __name__ == '__main__':
    rup = ResidualUpdatePolicy()
    rup.update(0, 1, 1)