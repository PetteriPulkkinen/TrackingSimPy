from trackingsimpy.tracking import TrackingComputer


class BaseRISimulation(object):
    def __init__(self, computer: TrackingComputer, k_min, k_max):
        self.target = computer.radar.target
        self.radar = computer.radar
        self.computer = computer
        self.k_min = k_min
        self.k_max = k_max

    def reset(self):
        self.target.reset()
        self.computer.initialize(x0=self.target.x)

    def step(self, revisit_interval):
        for _ in range(revisit_interval):
            self.computer.predict()
            trajectory_ends = self.target.trajectory_ends()
            if not trajectory_ends:
                self.target.update()
            else:
                break

        update_successful, n_missed = self.computer.update_track()

        return update_successful, n_missed, trajectory_ends
