from trackingsimpy.target import GenericTargetProcess


class DefinedTargetProcess(GenericTargetProcess):
    def __init__(self, x0, st_models, p_noises, order, dim):
        super(DefinedTargetProcess, self).__init__(x0, None, None, order=order, dim=dim)
        self.st_models = st_models
        self.p_noises = p_noises
        self._idx = 0

    def reset(self):
        super().reset()
        self._idx = 0
        self.F = None
        self.Q = None

    def update(self):
        self.F = self.st_models.get(self._idx, self.F)
        self.Q = self.p_noises.get(self._idx, self.Q)
        self._idx += 1
        return super().update()
