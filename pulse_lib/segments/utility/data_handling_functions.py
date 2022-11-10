from pulse_lib.segments.utility.looping import loop_obj
from pulse_lib.segments.data_classes.data_generic import data_container
from pulse_lib.segments.utility.setpoint_mgr import setpoint
from functools import wraps
import numpy as np
import copy

def find_common_dimension(dim_1, dim_2):
    '''
    finds the union of two dimensions
    Args:
        dim_1 (list/tuple) : list with dimensions of a first data object
        dim_2 (list/tuple) : list with dimensions of a second data object
    Returns:
        dim_comb (tuple) : the common dimensions of the of both dimensions provided

    Will raise error is dimensions are not compatible
    '''
    if dim_2 == (1,):
        return dim_1
    if dim_1 == dim_2:
        return dim_1
    if dim_1 == (1,):
        return dim_2

    try:
        res = np.broadcast_shapes(dim_1, dim_2)
        return res
    except:
        raise ValueError(f"Cannot combine dimensions of data objects {dim_1} and {dim_2}."
                         "This error is most likely caused by looping over two variables with different dimensions along the same axis.")


def update_dimension(data, new_dimension_info):
    '''
    update dimension of the data object to the one specified in new dimension_info
    Args:
        data (np.ndarray[dtype = object]) : numpy object that contains all the segment data of every iteration.
        new_dimension_info (list/np.ndarray) : list of the new dimensions of the array
    Returns:
        data (np.ndarray[dtype = object]) : same as input data, but with new_dimension_info.
    '''
    if data.shape == tuple(new_dimension_info):
        return data

    new_dimension_info = np.array(new_dimension_info)

    for i in range(len(new_dimension_info)):
        if data.ndim < i+1:
            data = _add_dimensions(data, new_dimension_info[-i-1:])

        elif list(data.shape)[-i -1] != new_dimension_info[-i -1]:
            shape = list(data.shape)
            shape[-i-1] = new_dimension_info[-i-1]
            data = _extend_dimensions(data, shape, -i-1)

    return data

def _add_dimensions(data, shape):
    """
    Function that can be used to add and extra dimension of an array object. A seperate function is needed since we want to make a copy and not a reference.
    Note that only one dimension can be extended!
    Args:
        data (np.ndarray[dtype = object]) : numpy object that contains all the segment data of every iteration.
        shape (list/np.ndarray) : list of the new dimensions of the array
    """
    new_data =  data_container(shape = shape)
    for i in range(shape[0]):
        new_data[i] = _cpy_numpy_shallow(data)
    return new_data

def _extend_dimensions(data, shape, new_axis):
    '''
    Extends the dimensions of a existing array object. This is useful if one would have first defined sweep axis 2 without defining axis 1.
    In this case axis 1 is implicitly made, only harbouring 1 element.
    This function can be used to change the axis 1 to a different size.

    Args:
        data (np.ndarray[dtype = object]) : numpy object that contains all the segment data of every iteration.
        shape (list/np.ndarray) : list of the new dimensions of the array (should have the same lenght as the dimension of the data!)
        axis (int): the axis added in shape
    '''
    new_data = data_container(shape=shape)

    if new_axis == 0:
        for j in range(len(new_data)):
            new_data[j] = _cpy_numpy_shallow(data)
    else:
        new_data = new_data.swapaxes(new_axis, 0)
        data = data.swapaxes(new_axis, 0)

        for j in range(len(new_data)):
            new_data[j] = _cpy_numpy_shallow(data)

        new_data = new_data.swapaxes(new_axis, 0)

    return new_data


def _cpy_numpy_shallow(data):
    '''
    Makes a shallow copy of an numpy object array.

    Args:
        data : data element
    '''
    if type(data) != data_container:
        return copy(data)

    if data.shape == (1,):
        return data[0].__copy__()

    shape = data.shape
    data_flat = data.flatten()
    new_arr = np.empty(data_flat.shape, dtype=object)

    for i in range(len(new_arr)):
        new_arr[i] = copy.copy(data_flat[i])

    new_arr = new_arr.reshape(shape)
    return new_arr


def _get_new_dim_loop(current_dim, axis, shape):
    '''
    function to get new dimensions from a loop spec.

    Args:
        current_dim [tuple/array] : current dimensions of the data object.
        axis [int] : on which axis to put the new loop dimension.
        shape [int] : the number of elements that a are along that loop axis.

    Returns:
        new_dim [array] : new dimensions of the data obeject when one would include the loop spec
        axis [int] : axis on which a loop variable was put (if free assign option was used (axis of -1))
    '''
    current_dim = list(current_dim)
    new_dim = []
    if axis == -1:
        # assume if last dimension has size 1, that you want to extend this direction.
        if current_dim[-1] == 1:
            new_dim = current_dim
            new_dim[-1] = shape
            axis = len(new_dim) - 1
        else:
            new_dim = [shape] + current_dim
            # assign new axis.
            axis = len(new_dim) - 1
    else:
        if axis >= len(current_dim):
            new_dim = [1]*(axis+1)
            for i in range(len(current_dim)):
                new_dim[axis-len(current_dim)+1 + i] = current_dim[i]
            new_dim[0] = shape
        else:
            if current_dim[-1-axis] == shape:
                new_dim = current_dim
            elif current_dim[-1-axis] == 1:
                new_dim = current_dim
                new_dim[-1-axis] = shape
            else:
                raise ValueError("Dimensions on loop axis {} not compatible with previous loops\n\
                    (current dimensions is {}, wanted is {}).\n\
                    Please change loop axis or update the length.".format(axis,
                    current_dim[-axis-1], shape))

    return tuple(new_dim), axis


def _update_segment_dims(segment, lp, arg_index, rendering=False):
    axes = list(lp.axis)
    data = segment.data if not rendering else segment.pulse_data_all
    for i in range(len(lp.axis)-1,-1,-1):

        data_shape = data.shape
        lp_axis = lp.axis[i]
        lp_length = lp.shape[i]
        new_shape, axis = _get_new_dim_loop(data_shape, lp_axis, lp_length)
        if new_shape != data_shape:
            if segment.is_slice:
                # TODO: Fix this with refactored indexing.
                raise Exception(f'Cannot resize data in slice (Indexing). '
                                'All loop axes must be added before indexing segment.')

        axes[i] = axis
        data = update_dimension(data, new_shape)

        if not lp.no_setpoints and lp.setvals is not None:
            sp = setpoint(axis, label=(lp.labels[i],), unit=(lp.units[i],), setpoint=(lp.setvals[i],))
            segment._setpoints += sp

    if not rendering:
        segment.data = data
    else:
        segment._pulse_data_all = data

    return {'arg_index':arg_index, 'axes':axes}

_in_loop = False
def loop_controller(func):
    '''
    Checks if there are there are parameters given that are loopable.

    If loop:
        * then check how many new loop parameters on which axis
        * extend data format to the right shape (simple python list used).
        * loop over the data and add called function

    if no loop, just apply func on all data (easy)
    '''
    @wraps(func)
    def wrapper(obj, *args, **kwargs):
        global _in_loop
        if _in_loop:
            raise Exception('NESTED LOOPS')
        _in_loop = True

        loop_info_args = []
        loop_info_kwargs = []

        for i,arg in enumerate(args):
            if isinstance(arg, loop_obj):
                loop_info = _update_segment_dims(obj, arg, i)
                loop_info_args.append(loop_info)

        for key,kwarg in kwargs.items():
            if isinstance(kwarg, loop_obj):
                loop_info = _update_segment_dims(obj, kwarg, key)
                loop_info_kwargs.append(loop_info)

        data = obj.data

        if len(loop_info_args) == 0 and len(loop_info_kwargs) == 0:
            if data.shape != (1,):
                loop_over_data(func, obj, data, args, kwargs)
            else:
                obj.data_tmp = data[0]
                data[0] = func(obj, *args, **kwargs)
        else:
            loop_over_data_lp(func, obj, data, args, loop_info_args, kwargs, loop_info_kwargs)
        _in_loop = False

    return wrapper


def loop_controller_post_processing(func):
    '''
    Checks if there are there are parameters given that are loopable.

    If loop:
        * then check how many new loop parameters on which axis
        * extend data format to the right shape (simple python list used).
        * loop over the data and add called function

    loop controller that works on the *pulse_data_all* object. This acts just before rendering. When rendering is done, all the actions of this looper are done.
    '''
    @wraps(func)
    def wrapper(obj, *args, **kwargs):

        loop_info_args = []
        loop_info_kwargs = []

        for i,arg in enumerate(args):
            if isinstance(arg, loop_obj):
                loop_info = _update_segment_dims(obj, arg, i, rendering=True)
                loop_info_args.append(loop_info)

        for key,kwarg in kwargs.items():
            if isinstance(kwarg, loop_obj):
                loop_info = _update_segment_dims(obj, kwarg, key, rendering=True)
                loop_info_kwargs.append(loop_info)

        data = obj.pulse_data_all
        if len(loop_info_args) > 0 or len(loop_info_kwargs) > 0:
            loop_over_data_lp(func, obj, data, args, loop_info_args, kwargs, loop_info_kwargs)
        else:
            loop_over_data(func, obj, data, args, kwargs)

    return wrapper

def loop_over_data_lp(func, obj, data, args, args_info, kwargs, kwargs_info):
    '''
    Recursive function to apply the func to data with looping args

    Args:
        func : function to execute
        obj: segment function is called on
        data : data of the segment
        args: arugments that are provided
        args_info : argument info is provided (e.g. axis updates)
        kwargs : kwargs provided
        kwarfs_info : same as args_info
    '''
    shape = data.shape
    n_dim = len(shape)

    # copy the input --> we will fill in the arrays
    # only copy when there are loops
    if len(args_info) > 0:
        # copy to new list
        args_cpy = list(args)
    else:
        args_cpy = args
    if len(kwargs_info) > 0:
        kwargs_cpy = kwargs.copy()
    else:
        kwargs_cpy = kwargs

    for i in range(shape[0]):
        for arg in args_info:
            if n_dim-1 in arg['axes']:
                index = arg['arg_index']
                args_cpy[index] = args[index][i]
        for kwarg in kwargs_info:
            if n_dim-1 in kwarg['axes']:
                index = kwarg['arg_index']
                kwargs_cpy[index] = kwargs[index][i]

        if n_dim == 1:
            # we are at the lowest level of the loop.
            obj.data_tmp = data[i]
            data[i] = func(obj, *args_cpy, **kwargs_cpy)
        else:
            # clean up args, kwargs
            loop_over_data_lp(func, obj, data[i], args_cpy, args_info, kwargs_cpy, kwargs_info)


def loop_over_data(func, obj, data, args, kwargs):
    '''
    Recursive function to apply func to data

    Args:
        func : function to execute
        obj: segment function is called on
        data : data of the segment
        args: arugments that are provided
        kwargs : kwargs provided
    '''
    shape = data.shape
    n_dim = len(shape)

    for i in range(shape[0]):

        if n_dim == 1:
            # we are at the lowest level of the loop.
            obj.data_tmp = data[i]
            data[i] = func(obj, *args, **kwargs)
        else:
            loop_over_data(func, obj, data[i], args, kwargs)


def reduce_arr(arr):
    """
    Return which elements on which axis are unique

    Args:
        arr (np.ndarray) : input array which to reduce to unique value

    Returns:
        reduced array(np.ndarray) : array with reduced data.
        data_axis (list) : the axises that have changing data.
    """
    shape = arr.shape
    ndim = len(shape)
    data_axis = []
    slice_array = ()
    for i in range(ndim):
        if shape[i] == 1:
            slice_array += (0,)
            continue
        mn = np.min(arr, axis=i)
        mx = np.max(arr, axis=i)
        eq = np.all(mn == mx)
        if not eq:
            data_axis.append(ndim - i - 1)
            slice_array += (slice(None),)
        else:
            slice_array += (0,)
    red_ar = arr[slice_array]
    return red_ar, data_axis
