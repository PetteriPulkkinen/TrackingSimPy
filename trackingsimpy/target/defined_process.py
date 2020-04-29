from trackingsimpy.target import GenericTargetProcess


class DefinedTargetProcess(GenericTargetProcess):
    def __init__(self, x0, st_models, p_noises, max_steps, order, dim):
        """

        :param x0: Initial state
        :param st_models: State transition models as dictionary where key is the index when to change
        :param p_noises: Similar to st_models but for process noise
        :param max_steps: Maximum number of update steps that can be done
        :param order:
        :param dim:
        """
        super(DefinedTargetProcess, self).__init__(x0, None, None, order=order, dim=dim)
        self.st_models = st_models
        self.p_noises = p_noises
        self._idx = 0
        self.max_steps = max_steps

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

    def trajectory_ends(self):
        return self._idx >= self.max_steps


