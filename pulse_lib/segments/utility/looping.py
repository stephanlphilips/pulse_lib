import numpy as np
import copy

class loop_obj():
    """object that initializes some standard fields that need to be there in a loop object.
    Args:
        no_setpoints (bool): ignore setpoints if False. Required for internal loop objects, e.g. as used in reset_time()
    """
    def __init__(self, no_setpoints=False):
        self.no_setpoints = no_setpoints
        # little inspiration from qcodes parameter ...
        self.labels = list()
        self.units = list()
        self.axis = list()
        self.dtype = None
        self.setvals = None
        self.setvals_set = False

    def add_data(self, data, axis = None, labels = None, units = None, setvals = None):
        '''
        add data to the loop object.
        data (array/np.ndarray) : n dimensional array with a regular shape (any allowed by numpy) of values you want to sweep
        axis (int/tuple<int>) : loop axis, if none is provided, a new loop axis is generated when the loop object is used in the pulse library.
        labels (str/tuple<str>) : name of the data that is swept. This will for example be used as an axis label
        units (str/tuple<str>) : unit of the sweep data
        setvals (array/np.ndarray) : if you want to display different things on the axis than the normal data point. When None, setvals is the same as the data varaible.
        '''
        self.data = np.asarray(data)
        self.dtype = self.data.dtype

        if axis is None:
            self.axis = [-1]*len(self.data.shape)
        elif type(axis) == int:
            self.axis = [axis]
        else:
            if len(axis) != len(self.data.shape):
                raise ValueError(f"Provided incorrect dimensions for the axis (axis:{axis} <> data:{self.data.shape})")
            self.axis = axis

        if labels is None:
            self.labels = tuple(['no_label']*self.ndim)
        elif type(labels) == str:
            self.labels = (labels, )
        else:
            if len(labels) != len(self.data.shape):
                raise ValueError("Provided incorrect dimensions for the axis.")
            self.labels = labels

        if units is None:
            self.units = tuple(['no_label']*self.ndim)
        elif type(units) == str:
            self.units = (units, )
        else:
            if len(units) != len(self.data.shape):
                raise ValueError("Provided incorrect dimensions for the axis.")
            self.units = units

        if not self.no_setpoints:
            if setvals is None:
                if len(data.shape) == 1:
                    self.setvals = (self.data, )
                else:
                        raise ValueError ('Multidimensional setpoints cannot be inferred from input.')
            else:
                self.setvals = tuple()
                if isinstance(setvals,list):
                    setvals = np.asarray(setvals)

                    if self.shape != setvals.shape:
                        raise ValueError("setvals should have the same dimensions as the data dimensions.")
                    setvals = (setvals, )
                else:
                    setvals = list(setvals)
                    for setval_idx in range(len(setvals)):
                        setvals[setval_idx] = np.asarray(setvals[setval_idx])

                        if self.shape[setval_idx] != len(setvals[setval_idx]):
                            raise ValueError("setvals should have the same dimensions as the data dimensions.")

                    setvals = tuple(setvals)

                self.setvals += setvals
                self.setvals_set = True

    def __len__(self):
        return len(self.data)

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    def __getitem__(self, key):
        if self.ndim == 1:
            return self.data[key]
        else:
            partial = loop_obj()
            partial.labels =self.labels[1:]
            partial.units = self.units[1:]
            partial.axis = self.axis[1:]
            partial.dtype = self.dtype
            partial.data = self.data[key]
            return partial

    def __add__(self, other):
        cpy = copy.copy(self)
        cpy.data += other
        return cpy

    def __radd__(self, other):
        cpy = copy.copy(self)
        cpy.data += other
        return cpy

    def __mul__(self, other):
        cpy = copy.copy(self)
        cpy.data *= other
        return cpy

    def __rmul__(self, other):
        cpy = copy.copy(self)
        cpy.data *= other
        return cpy

    def __sub__(self, other):
        cpy = copy.copy(self)
        cpy.data -= other
        return cpy

    def __rsub__(self, other):
        cpy = copy.copy(self)
        cpy.data = other - cpy.data
        return cpy

    def __truediv__(self, other):
        cpy = copy.copy(self)
        cpy.data += self.data/other
        return cpy

    def __copy__(self):
        cpy = loop_obj()
        cpy.labels = copy.copy(self.labels)
        cpy.setvals = copy.copy(self.setvals)
        cpy.setvals_set = copy.copy(self.setvals_set)
        cpy.units = copy.copy(self.units)
        cpy.axis = copy.copy(self.axis)
        cpy.dtype = copy.copy(self.dtype)

        if hasattr(self, 'data'):
            cpy.data= copy.copy(self.data)
        return cpy


class linspace(loop_obj):
    """docstring for linspace"""
    def __init__(self, start, stop, n_steps = 50, name = None, unit = None, axis = -1, setvals = None):
        super().__init__()
        super().add_data(np.linspace(start, stop, n_steps), axis = axis, labels = name, units = unit, setvals= setvals)

class logspace(loop_obj):
    """docstring for logspace"""
    def __init__(self, start, stop, n_steps = 50, name = None, unit = None, axis = -1, setvals = None):
        super().__init__()
        super().add_data(np.logspace(np.log10(start), np.log10(stop), n_steps), axis = axis, labels = name, units = unit, setvals= setvals)

class geomspace(loop_obj):
    """docstring for geomspace"""
    def __init__(self, start, stop, n_steps = 50, name = None, unit = None, axis = -1, setvals = None):
        super().__init__()
        super().add_data(np.geomspace(start, stop, n_steps), axis = axis, labels = name, units = unit, setvals= setvals)

if __name__ == '__main__':
    lp = loop_obj()
    lp.add_data(np.array([1]), axis=0)
    print(lp.labels)

    data = np.linspace(0,5,10)
    lp = loop_obj()
    lp.add_data(data, axis=0, labels = "gate_name", units = 'mV')
    print(lp.data)
    print(lp.axis)
    print(lp.labels)
    print(lp.units)
    print(lp.setvals)

    data = np.zeros([4,4])
    lp = loop_obj()
    lp.add_data(data, axis=[0,1], labels = ("gate_name_1", "gate_name_2"), units = ('mV', 'mV'), setvals=([0,0,0,0],[1,2,3,4]))
    print(lp.data)
    print(lp.axis)
    print(lp.labels)
    print(lp.units)
    print(lp.setvals)

    lp = linspace(0, 10, 5)

    lp2 = lp + 100
    print(lp2.setvals)
    print(lp2.data)

    lp2 = 100 + lp
    print(lp2.data)

    lp2 = lp - 100
    print(lp2.data)

    lp2 = 100 - lp
    print(lp2.data)

    lp2 = lp * 100
    print(lp2.data)

    lp2 = 100 * lp
    print(lp2.data)
