class Environment(object):
    def __init__(self):
        self.targets = dict()
        self.radars = dict()

    def update(self):
        for target in self.targets.values():
            target.update()

        for radar in self.radars.values():
            radar.update()
