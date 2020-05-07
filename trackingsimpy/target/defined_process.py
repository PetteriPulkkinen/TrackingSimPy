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
        super(DefinedTargetProcess, self).__init__(x0, list(st_models.values())[0], list(p_noises.values())[0],
                                                   max_steps, order=order, dim=dim)
        self.st_models = st_models
        self.p_noises = p_noises
        self.max_steps = max_steps

    def update(self):
        F = self.st_models.get(self._idx)
        Q = self.p_noises.get(self._idx)

        if F is not None:
            self.F = F
        if Q is not None:
            self.Q = Q

        return super().update()

