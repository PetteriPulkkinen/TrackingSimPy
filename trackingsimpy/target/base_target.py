class BaseTarget(object):
    def __init__(self, order, dim):
        self.order = order
        self.dim = dim
        self.x = None

    def update(self):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    @property
    def state(self):
        return self.x

    @property
    def position(self):
        idxs = [i * (self.order + 1) for i in range(self.dim)]
        return self.x.flatten()[idxs]
