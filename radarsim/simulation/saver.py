class AttributeSaver(object):
    """Class to save attributes of Python objects."""
    def __init__(self, obj, attr_list):
        """
        Args:
            obj: Python object whose attributes will be saved.
            attr_list: List or Tuple of attributes using python strings.
        """
        assert(isinstance(attr_list, (list, tuple)))

        self.object = obj
        self.arg_list = attr_list
        self._ds = dict()

        for arg in attr_list:
            self._ds[arg] = list()

    def __getitem__(self, key):
        return self._ds[key]

    def save(self):
        """
        Save all attributes.
        """
        for arg in self.arg_list:
            self[arg].append(getattr(self.object, arg))
