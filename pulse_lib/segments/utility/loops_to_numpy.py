import numpy as np
from functools import wraps
from copy import copy
from collections.abc import Iterable

from pulse_lib.segments.utility.looping import loop_obj


def to_numpy(obj:loop_obj):
    shape = [1] * (max(obj.axis)+1)
    for dim, l in zip(obj.axis, obj.shape):
        shape[dim] = l
    # obj.data is already a numpy object
    return obj.data.reshape(shape)

def select(tup, indices):
    return tuple(value for i,value in enumerate(tup) if i in indices)

def to_loop_obj(obj, joined_loops):
    res = copy(joined_loops)
    res_axis = []
    selected_loop_axis = []
    for idim, l in enumerate(obj.shape):
        if l == 1:
            continue
        if idim not in joined_loops.axis:
            raise Exception(f'Cannot convert {obj.shape} using axis {joined_loops.axis}')
        res_axis.append(idim)
        selected_loop_axis.append(joined_loops.axis.index(idim))

    print('to_loop:', obj)
    print(selected_loop_axis)
    res_axis.reverse()
    res.labels = select(res.labels, selected_loop_axis)
    res.setvals = select(res.setvals, selected_loop_axis)
    res.units = select(res.units, selected_loop_axis)
    res.axis = res_axis
    res.data = np.squeeze(obj)
    return res

def to_loop_objs(objs, loop_objs):
    joined_loops = sum(loop_objs)
    if isinstance(objs, Iterable):
        res = (to_loop_obj(obj, joined_loops) for obj in objs)
    else:
        res = to_loop_obj(objs, joined_loops)
    return res

def loops_to_numpy(func):
    '''
    Checks if there are there are parameters given that are loopable.

    If loop:
        * then check how many new loop parameters on which axis
        * extend data format to the right shape (simple python list used).
        * loop over the data and add called function

    if no loop, just apply func on all data (easy)
    '''
    @wraps(func)
    def wrapper(*args, **kwargs):
        loop_objs = []
        arg_list = list(args)
        for i in range(0,len(arg_list)):
            if isinstance(arg_list[i], loop_obj):
                loop_objs.append(arg_list[i])
                arg_list[i] = to_numpy(arg_list[i])

        for key in kwargs.keys():
            if isinstance(kwargs[key], loop_obj):
                loop_objs.append(kwargs[key])
                kwargs[key] = to_numpy(kwargs[key])

        res = func(*arg_list, **kwargs)
        return to_loop_objs(res, loop_objs)


    return wrapper

if __name__ == '__main__':
    from pulse_lib.segments.utility.looping import linspace
    from pulse_lib.base_pulse import pulselib
    import matplotlib.pyplot as plt

    @loops_to_numpy
    def triangle(height:np.ndarray, slope1:np.ndarray, slope2:np.ndarray) -> np.ndarray:
        '''
        height: height in mV
        slope1: rising slope in mV/ns
        slope2: falling slope in mV/ns
        '''
        print(slope1)
        print(slope2)
        t_ramp1 = height/slope1
        t_ramp2 = -height/slope2
        return t_ramp1, t_ramp2

    p = pulselib('')
    p.define_channel('P1', 'AWG1', 1)

    slope1 = linspace(10, 100, 10, axis=0)
    slope2 = -linspace(20, 100, 5, axis=1)
    height = 400

    t_ramp1, t_ramp2 = triangle(height, slope1, slope2)

    print(t_ramp1)
    print(t_ramp1.data)
    print(t_ramp2)
    print(t_ramp2.data)

    s = p.mk_segment()
    s.P1.add_ramp_ss(0, t_ramp1, 0, height)
    s.reset_time()
    s.P1.add_ramp_ss(0, t_ramp2, height, 0)

    s.plot([0,0])
    s.plot([0,5])
    s.plot([1,0])
