class BaseRadar(object):
    def __init__(self, target, dim, order):
        self.target = target
        self.dim = dim
        self.order = order

    def illuminate(self, prediction):
        raise NotImplementedError

    def set_target(self, target):
        self.target = target
