import numpy as np


class Saver(object):
    def __init__(self, structure: dict):
        self.structure = structure
        self.storage = dict()
        self.reset()

    def __getitem__(self, item):
        return self.storage[item]

    def convert_to_numpy(self):
        for (key, value) in self._iterate_structure():
            self.storage[key][value] = np.array(self.storage[key][value])

    def reset(self):
        for key, value in self._iterate_structure():
            if key not in self.storage:
                self.storage[key] = dict()
            self.storage[key][value] = list()

    def save(self):
        for (key, value) in self._iterate_structure():
            self.storage[key][value].append(getattr(key, value))

    def _iterate_structure(self):
        for (key, value) in self.structure.items():
            if isinstance(value, (list, tuple)):
                for attrib in value:
                    yield key, attrib
            elif isinstance(value, str):
                yield key, value
            else:
                raise RuntimeError("Data type not understood! Check the data structure of the saver.")
