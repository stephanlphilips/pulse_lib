import numpy as np
import copy

class loop_obj():
    """object that initializes some standard fields that need to be there in a loop object.
    Args:
        no_setpoints (bool): ignore setpoints if False. Required for internal loop objects, e.g. as used in reset_time()
    """
    def __init__(self, no_setpoints=False):
        self.no_setpoints = no_setpoints
        self.names = list()
        self.labels = list()
        self.units = list()
        self.axis = list()
        self.dtype = None
        self.setvals = None
        self.setvals_set = False

    def add_data(self, data, axis=None, names=None, labels=None, units=None, setvals=None,
                 dtype=None):
        '''
        add data to the loop object.
        data (array/np.ndarray) : n dimensional array with a regular shape (any allowed by numpy) of values you want to sweep
        axis (int/tuple<int>) : loop axis, if none is provided, a new loop axis is generated when the loop object is used in the pulse library.
        names (str/tuple<str>) : name of the data that is swept.
        labels (str/tuple<str>) : label of the data that is swept. This will for example be used as an axis label
        units (str/tuple<str>) : unit of the sweep data
        setvals (array/np.ndarray) : if you want to display different things on the axis than the normal data point. When None, setvals is the same as the data varaible.
        '''
        # NOTE: data MUST be float for qcodes save_metadata
        if dtype is not object:
            self.data = np.asarray(data).astype(float)
        else:
            self.data = np.asarray(data)
        self.dtype = self.data.dtype

        if axis is None:
            self.axis = [-1]*len(self.data.shape)
        elif type(axis) == int:
            self.axis = [axis]
        else:
            if len(axis) != len(self.data.shape):
                raise ValueError(f"Provided incorrect dimensions for the axis (axis:{axis} <> data:{self.data.shape})")
            if sorted(list(axis), reverse=True) != list(axis):
                raise ValueError("Axis must be defined in descending order, e.g. [1,0]")
            self.axis = axis

        if self.no_setpoints:
            return

        if names is None:
            names = labels
        elif labels is None:
            labels = names

        if names is None:
            self.names = tuple(['no_name']*self.ndim)
        elif type(names) == str:
            self.names = (names, )
        else:
            if len(names) != len(self.data.shape):
                raise ValueError("Provided incorrect names.")
            self.names = names

        if labels is None:
            self.labels = tuple(['no_label']*self.ndim)
        elif type(labels) == str:
            self.labels = (labels, )
        else:
            if len(labels) != len(self.data.shape):
                raise ValueError("Provided incorrect labels.")
            self.labels = labels

        if units is None:
            self.units = tuple(['no_label']*self.ndim)
        elif type(units) == str:
            self.units = (units, )
        else:
            if len(units) != len(self.data.shape):
                raise ValueError("Provided incorrect units.")
            self.units = units

        if setvals is None:
            if len(self.data.shape) == 1:
                self.setvals = (self.data, )
            else:
                raise ValueError ('Multidimensional setpoints cannot be inferred from input.')
        else:
            self.setvals = tuple()
            if isinstance(setvals,list) or isinstance(setvals, np.ndarray):
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

    def at(self, seg_index):
        '''
        Returns the value for this specific index in the segment.
        E.g. If axis = [0] and seg_index is (10,20,30) then
        it returns self[30].
        Note: In pulse-lib axis 0 is the right-most axis of the shape.
        '''
        key = tuple(seg_index[-i-1] for i in self.axis)
        return self.data[key]

    def __getitem__(self, key):
        if self.ndim == 1:
            return self.data[key]
        else:
            partial = loop_obj()
            partial.names = self.names[1:]
            partial.labels = self.labels[1:]
            partial.units = self.units[1:]
            partial.axis = self.axis[1:]
            partial.dtype = self.dtype
            partial.data = self.data[key]
            return partial

    def __add__(self, other):
        cpy = copy.copy(self)
        if isinstance(other, loop_obj):
            # combine axis returns the reshaped data
            cpy_data, other_data = loop_obj.__combine_axis(cpy, other)
            cpy.data = cpy_data + other_data
        else:
            cpy.data += other
        return cpy

    def __radd__(self, other):
        # only called if other is not loop_obj
        return self.__add__(other)

    def __mul__(self, other):
        cpy = copy.copy(self)
        if isinstance(other, loop_obj):
            cpy_data, other_data = loop_obj.__combine_axis(cpy, other)
            cpy.data = cpy_data * other_data
        else:
            cpy.data *= other
        return cpy

    def __rmul__(self, other):
        # only called if other is not loop_obj
        cpy = copy.copy(self)
        cpy.data *= other
        return cpy

    def __sub__(self, other):
        cpy = copy.copy(self)
        if isinstance(other, loop_obj):
            cpy_data, other_data = loop_obj.__combine_axis(cpy, other)
            cpy.data = cpy_data - other_data
        else:
            cpy.data -= other
        return cpy

    def __rsub__(self, other):
        # only called if other is not loop_obj
        cpy = copy.copy(self)
        cpy.data = other - cpy.data
        return cpy

    def __neg__(self):
        cpy = copy.copy(self)
        cpy.data = -cpy.data
        return cpy

    def __truediv__(self, other):
        cpy = copy.copy(self)
        if isinstance(other, loop_obj):
            cpy_data, other_data = loop_obj.__combine_axis(cpy, other)
            cpy.data = cpy_data / other_data
        else:
            cpy.data /= other
        return cpy

    def __rtruediv__(self, lhs):
        cpy = copy.copy(self)
        cpy.data = lhs / cpy.data
        return cpy

    def __round__(self, ndigits=None):
        cpy = copy.copy(self)
        cpy.data = np.round(self.data, ndigits)
        return cpy

    def __trunc__(self):
        cpy = copy.copy(self)
        cpy.data = np.trunc(self.data)
        return cpy

    def __floor__(self):
        cpy = copy.copy(self)
        cpy.data = np.floor(self.data)
        return cpy

    def __ceil__(self):
        cpy = copy.copy(self)
        cpy.data = np.ceil(self.data)
        return cpy

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        '''
        Called by numpy when numpy method is applied to loop_obj.
        Applies ufunc on self.data and returns new loop_obj.
        See numpy documentation.
        '''
        if inputs[0] is self and method == '__call__':
            cpy = copy.copy(self)
            if 'out' in kwargs:
                raise Exception('out not yet supported.')
            args = list(inputs[1:])
            if len(args) > 0 and isinstance(args[0], loop_obj):
                self_data, other_data = loop_obj.__combine_axis(cpy, args[0])
                cpy.data = ufunc(self_data, other_data, *args[1:], **kwargs)
            else:
                cpy.data = ufunc(self.data, *args, **kwargs)
            return cpy
        elif inputs[0] is self:
            return getattr(ufunc, method)(self.data, *inputs[1:], **kwargs)
        elif len(inputs) > 1 and inputs[1] is self and method == '__call__':
            return ufunc(inputs[0], self.data, *inputs[2:], **kwargs)
        return NotImplemented

    def __copy__(self):
        cpy = loop_obj()
        cpy.names = copy.copy(self.names)
        cpy.labels = copy.copy(self.labels)
        cpy.setvals = copy.copy(self.setvals)
        cpy.setvals_set = self.setvals_set
        cpy.no_setpoints = self.no_setpoints
        cpy.units = copy.copy(self.units)
        cpy.axis = copy.copy(self.axis)
        cpy.dtype = copy.copy(self.dtype)

        if hasattr(self, 'data'):
            cpy.data= copy.copy(self.data)
        return cpy

#    def _JSONEncoder(self):
#        klass = self.__class__
#        module = klass.__module__
#        if module and module != '__builtin__':
#            full_name = module + '.' + klass.__name__ # avoid outputs like '__builtin__.str'
#        else:
#            full_name = klass.__name__
#        d = dict(__dtype__=full_name)
#        d.update(self.to_dict())
#        return d

    def to_dict(self):
        return {
            'names': self.names,
            'labels': self.labels,
            'setvals': self.setvals,
            'units': self.units,
            'axis': self.axis,
            'dtype': self.dtype,
            'data': self.data,
            }

    @classmethod
    def from_dict(cls, data):
        res = loop_obj()
        res.add_data(**data)
        return res

    @staticmethod
    def __combine_axis(this, other):
        if this.axis == other.axis and this.shape == other.shape:
            return this.data, other.data

        new_axis = sorted(set(this.axis) | set(other.axis))
        new_axis.reverse()

        sel_this = []
        sel_other = []
        new_names = []
        new_labels = []
        new_units = []
        new_setvals = []
        for axis in new_axis:
            try:
                ithis = this.axis.index(axis)
                sel_this.append(slice(None))
                if not this.no_setpoints:
                    new_names.append(this.names[ithis])
                    new_labels.append(this.labels[ithis])
                    new_units.append(this.units[ithis])
                    new_setvals.append(this.setvals[ithis])
                try:
                    # check equality of shapes
                    iother = other.axis.index(axis)
                    if this.shape[ithis] != other.shape[iother]:
                        raise Exception(f'Cannot combine loops {this} and {other}. '
                                        'Shapes do not match')
                except ValueError:
                    pass
            except ValueError:
                # add new axis
                sel_this.append(np.newaxis)
                iother = other.axis.index(axis)
                if not this.no_setpoints:
                    new_names.append(other.names[iother])
                    new_labels.append(other.labels[iother])
                    new_units.append(other.units[iother])
                    new_setvals.append(other.setvals[iother])
            try:
                iother = other.axis.index(axis)
                sel_other.append(slice(None))
            except ValueError:
                sel_other.append(np.newaxis)

        this.axis = new_axis
        if not this.no_setpoints:
            this.names = tuple(new_names)
            this.labels = tuple(new_labels)
            this.units = tuple(new_units)
            this.setvals = tuple(new_setvals)

        return this.data[tuple(sel_this)], other.data[tuple(sel_other)]

    def __repr__(self):
        return (
            f'loop(names: {self.names}, axis:{self.axis}, labels:{self.labels}, units: {self.units}, '
            f'setvals: {self.setvals}, data: {self.data})'
            )


class linspace(loop_obj):
    """docstring for linspace"""
    def __init__(self, start, stop, n_steps=50,
                 name=None, label=None, unit=None, axis=-1, setvals=None,
                 endpoint=True):
        super().__init__()
        super().add_data(np.linspace(start, stop, n_steps, endpoint=endpoint),
                         axis=axis, names=name, labels=label, units=unit, setvals=setvals)

class logspace(loop_obj):
    """docstring for logspace"""
    def __init__(self, start, stop, n_steps=50,
                 name=None, label=None, unit=None, axis=-1, setvals=None,
                 endpoint=True):
        super().__init__()
        super().add_data(np.logspace(start, stop, n_steps, endpoint=endpoint),
                         axis=axis, names=name, labels=label, units=unit, setvals=setvals)

class geomspace(loop_obj):
    """docstring for geomspace"""
    def __init__(self, start, stop, n_steps = 50,
                 name=None, label=None, unit=None, axis=-1, setvals=None,
                 endpoint=True):
        super().__init__()
        super().add_data(np.geomspace(start, stop, n_steps, endpoint=endpoint),
                         axis=axis, names=name, labels=label, units=unit, setvals=setvals)

class arange(loop_obj):
    def __init__(self, start, stop=None, step=1,
                 name=None, label=None, unit=None, axis=-1, setvals=None):
        super().__init__()
        if stop is None:
            stop = start
            start = 0
        super().add_data(np.arange(start, stop, step),
                         axis=axis, names=name, labels=label, units=unit, setvals=setvals)

class array(loop_obj):
    def __init__(self, data,
                 name=None, label=None, unit=None, axis=-1, setvals=None):
        super().__init__()
        super().add_data(data,
                         axis=axis, names=name, labels=label, units=unit, setvals=setvals)


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

    data = np.arange(10,40,10)[:,np.newaxis] + np.arange(1,5)[np.newaxis,:]
    lp = loop_obj()
    lp.add_data(data, axis=[1,0], labels = ("gate_name_1", "gate_name_2"), units = ('mV', 'mV'), setvals=([10,20,30],[1,2,3,4]))

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
