import numpy as np


class BaseUpdatePolicy(object):
    def __init__(self):
        self._K = 1
        self._counter = 0

    def is_revisit(self):
        return (self._counter % self._K) == 0

    def update(self, tracker, measurement):
        raise NotImplementedError

    def roll_forward(self):
        self._counter += 1

    def get_revisit_interval(self):
        return self._K

    def reset(self):
        self._counter = 0


class RandomUpdatePolicy(BaseUpdatePolicy):
    def __init__(self, intervals):
        super(RandomUpdatePolicy, self).__init__()
        self.intervals = intervals
        self.update(None, None)

    def update(self, tracker, measurement):
        self._K = np.random.choice(self.intervals)
        
    def reset(self):
        super().reset()
        self.update(None, None)
    
    
class ConstantUpdatePolicy(BaseUpdatePolicy):
    def __init__(self, interval):
        super(ConstantUpdatePolicy, self).__init__()
        self.interval = interval
        self.update(None, None)
        
    def update(self, tracker, measurement):
        self._K = self.interval

    def reset(self):
        super().reset()
        self.update(None, None)


class ResidualUpdatePolicy(BaseUpdatePolicy):
    def __init__(self, v0=1, K_max=10):
        super(ResidualUpdatePolicy, self).__init__()
        self.v0 = v0
        self.K_max = K_max

    def update(self, tracker, measurement):
        est_pos = (tracker.H @ tracker.x).flatten()
        distance = np.linalg.norm(est_pos - measurement.z.flatten())
        w, _ = np.linalg.eig(measurement.R_est)
        sig_th = np.sqrt(np.max(w))
        self._K = np.max([1, np.round(self._K*np.power(self.v0*sig_th/distance, 1/3))])
        self._K = np.min([self._K, self.K_max])

    def reset(self):
        super().reset()
        self._K = 1
