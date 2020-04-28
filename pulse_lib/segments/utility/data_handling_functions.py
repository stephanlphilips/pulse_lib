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
    dim_1 = list(dim_1)[::-1]
    dim_2 = list(dim_2)[::-1]
    dim_comb = []

    # estimate size of new dimension
    n_dim = len(dim_1)
    if len(dim_2) > n_dim:
        n_dim = len(dim_2)

    # combine both
    for i in range(n_dim):
        if len(dim_2) <= i:
            dim_comb.append(dim_1[i])
        elif len(dim_1) <= i:
            dim_comb.append(dim_2[i])
        else:
            if dim_1[i] == dim_2[i]:
                dim_comb.append(dim_2[i])
            elif dim_1[i] == 1:
                dim_comb.append(dim_2[i])
            elif dim_2[i] == 1:
                dim_comb.append(dim_1[i])
            else:
                raise ValueError("Error in combining dimensions of two data objects. This error is most likely caused by looping over two variables with different dimensions along the same axis.")

    return tuple(dim_comb[::-1])

def update_dimension(data, new_dimension_info, use_ref = False):
    '''
    update dimension of the data object to the one specified in new dimension_info
    Args:
        data (np.ndarray[dtype = object]) : numpy object that contains all the segment data of every iteration.
        new_dimension_info (list/np.ndarray) : list of the new dimensions of the array
        use_ref (bool) : use pointer to copy, or take full copy (False is full copy)
    Returns:
        data (np.ndarray[dtype = object]) : same as input data, but with new_dimension_info.
    '''

    new_dimension_info = np.array(new_dimension_info)

    for i in range(len(new_dimension_info)):
        if data.ndim < i+1:
            data = _add_dimensions(data, new_dimension_info[-i-1:], use_ref)

        elif list(data.shape)[-i -1] != new_dimension_info[-i -1]:
            shape = list(data.shape)
            shape[-i-1] = new_dimension_info[-i-1]
            data = _extend_dimensions(data, shape, use_ref)

    return data

def _add_dimensions(data, shape, use_ref):
    """
    Function that can be used to add and extra dimension of an array object. A seperate function is needed since we want to make a copy and not a reference.
    Note that only one dimension can be extended!
    Args:
        data (np.ndarray[dtype = object]) : numpy object that contains all the segment data of every iteration.
        shape (list/np.ndarray) : list of the new dimensions of the array
        use_ref (bool) : use pointer to copy, or take full copy (False is full copy)
    """
    new_data =  data_container(shape = shape)
    for i in range(shape[0]):
        new_data[i] = cpy_numpy_shallow(data, use_ref)
    return new_data

def _extend_dimensions(data, shape, use_ref):
    '''
    Extends the dimensions of a existing array object. This is useful if one would have first defined sweep axis 2 without defining axis 1.
    In this case axis 1 is implicitly made, only harbouring 1 element.
    This function can be used to change the axis 1 to a different size.

    Args:
        data (np.ndarray[dtype = object]) : numpy object that contains all the segment data of every iteration.
        shape (list/np.ndarray) : list of the new dimensions of the array (should have the same lenght as the dimension of the data!)
        use_ref (bool) : use pointer to copy, or take full copy (False is full copy)
    '''
    new_data = data_container(shape = shape)
    for i in range(len(shape)):
        if data.shape[i] != shape[i]:
            if i == 0:
                for j in range(len(new_data)):
                    new_data[j] = cpy_numpy_shallow(data, use_ref)
            else:
                new_data = new_data.swapaxes(i, 0)
                data = data.swapaxes(i, 0)

                for j in range(len(new_data)):
                    new_data[j] = cpy_numpy_shallow(data, use_ref)

                new_data = new_data.swapaxes(i, 0)


    return new_data


def cpy_numpy_shallow(data, use_ref):
    '''
    Makes a shallow copy of an numpy object array.

    Args:
        data : data element
        use_ref (bool) : use reference to copy
    '''

    if use_ref == True:
        if type(data) != data_container:
            return data

        if data.shape == (1,):
            return data[0]

        shape = data.shape
        data_flat = data.flatten()
        new_arr = np.empty(data_flat.shape, dtype=object)

        for i in range(len(new_arr)):
            new_arr[i] = data_flat[i]

        new_arr = new_arr.reshape(shape)

    else:
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


def _get_loop_info(lp, index):
    if lp.no_setpoints or lp.setvals is None:
        setpnt = None
    else:
        setpnt=list()

        for j in range(len(lp.axis)):
            setpnt_single = setpoint(lp.axis[j], label = (lp.labels[j],), unit = (lp.units[j],), setpoint=(lp.setvals[j],))
            setpnt.append(setpnt_single)

    info = {
    'nth_arg': index,
    'shape' : lp.shape,
    'len': len(lp),
    'axis': lp.axis,
    'data' : lp.data,
    'setpnt' : setpnt
    }
    return info


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
    def wrapper(*args, **kwargs):
        obj = args[0]

        loop_info_args = []
        loop_info_kwargs = []
        for i in range(1,len(args)):
            if isinstance(args[i], loop_obj):
                loop_info_args.append(_get_loop_info(args[i], i))

        for key in kwargs.keys():
            if isinstance(kwargs[key], loop_obj):
                loop_info_kwargs.append(_get_loop_info(kwargs[key], key))

        for lp in loop_info_args:
            for i in range(len(lp['axis'])-1,-1,-1):
                new_dim, axis = get_new_dim_loop(obj.data.shape, lp['axis'][i], lp['shape'][i])
                lp['axis'][i] = axis
                obj.data = update_dimension(obj.data, new_dim)

                if lp['setpnt'] is not None:
                    lp['setpnt'][i].axis = axis
                    obj._setpoints += lp['setpnt'][i]

        # todo update : (not used atm, but just to be generaric.)
        for lp in loop_info_kwargs:
            new_dim = get_new_dim_loop(obj.data.shape, lp)
            obj.data = update_dimension(obj.data, new_dim)

        loop_over_data(func, obj.data, args, loop_info_args, kwargs, loop_info_kwargs)


    return wrapper


def loop_controller_post_processing(func):
    '''
    Checks if there are there are parameters given that are loopable.

    If loop:
        * then check how many new loop parameters on which axis
        * extend data format to the right shape (simple python list used).
        * loop over the data and add called function

    loop controller that works on the pulse_data_all object. This acts just before rendering. When rendering is done, all the actions of this looper are done.
    '''
    @wraps(func)
    def wrapper(*args, **kwargs):
        obj = args[0]

        loop_info_args = []
        loop_info_kwargs = []

        for i in range(1,len(args)):
            if isinstance(args[i], loop_obj):
                loop_info_args.append(_get_loop_info(args[i], i))

        for key in kwargs.keys():
            if isinstance(kwargs[key], loop_obj):
                loop_info_kwargs.append(_get_loop_info(kwargs[key], key))

        for lp in loop_info_args:
            for i in range(len(lp['axis'])-1,-1,-1):
                new_dim, axis = get_new_dim_loop(obj.pulse_data_all.shape, lp['axis'][i], lp['shape'][i])
                lp['axis'][i] = axis
                obj._pulse_data_all = update_dimension(obj.pulse_data_all, new_dim)

                if lp['setpnt'] is not None:
                    lp['setpnt'][i].axis = axis
                    obj._setpoints += lp['setpnt'][i]

        # todo update : (not used atm, but just to be generaric.)
        for lp in loop_info_kwargs:
            new_dim = get_new_dim_loop(obj.pulse_data_all.shape, lp)
            obj.pulse_data_all = update_dimension(obj.pulse_data_all, new_dim)
        loop_over_data(func, obj.pulse_data_all, args, loop_info_args, kwargs, loop_info_kwargs)


    return wrapper



def loop_over_data(func, data, args, args_info, kwargs, kwargs_info):
    '''
    recursive function to apply the

    Args:
        func : function to execute
        data : data of the segment
        args: arugments that are provided
        args_info : argument info is provided (e.g. axis updates)
        kwargs : kwargs provided
        kwarfs_info : same as args_info
        loop_dimension


    Returns:
        None
    '''
    shape = list(data.shape)
    n_dim = len(shape)

    # copy the input --> we will fill in the arrays
    args_cpy = list(copy.copy(args))
    kwargs_cpy = copy.copy(kwargs)

    for i in range(shape[0]):
        for arg in args_info:
            if n_dim-1 in arg['axis']:
                args_cpy[arg['nth_arg']] = args[arg['nth_arg']][i]
        for kwarg in kwargs_info:
            if n_dim-1 in kwarg['axis']:
                kwargs_cpy[kwargs_info['nth_arg']] = kwargs[kwargs_info['nth_arg']][i]

        if n_dim == 1:
            # we are at the lowest level of the loop.
            args_cpy[0].data_tmp = data[i]
            data[i] = func(*args_cpy, **kwargs_cpy)
        else:
            # clean up args, kwargs
            loop_over_data(func, data[i], args_cpy, args_info, kwargs_cpy, kwargs_info)

def get_new_dim_loop(current_dim, axis, shape):
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

    return new_dim, axis

def update_labels(segment, loop_spec):
    """
    Update the labels in the current segment.

    Args:
        segment (segment_single) : segment object of which to update the labels
        loop_spec (dict) : dict containing the loop specification, with the updata data.
    """
    units = ['a.u.']*segment.ndim
    names = ['undefined']*segment.ndim
    setvals  = []*segment.ndim

    units[:len(segment._units)] = segment._units
    names[:len(segment._units)] = segment._names
    setvals[:len(segment._units)] = segment._setvals

    for loop in loop_spec:
        for i in range(len(loop['axis'])):
            if loop['data_obj'].units[i] != 'a.u.':
                units[loop['axis'][i]] = loop['data_obj'].units[i]
            if loop['data_obj'].names[i] != 'undefined':
                names[loop['axis'][i]] = loop['data_obj'].names[i]
            setvals[loop['axis'][i]] = loop['data_obj'].setvals[i]


    segment._units = units

def get_union_of_shapes(shape1, shape2):
    """
    function that combines the shape of two shapes.
    Args:
        shape1 (tuple/array) : shape of the first array
        shape2 (tuple/array) : shape of the second array

    Returns
        new_shape (tuple) : shape of the combined array
    """
    shape1 = list(shape1)
    shape2 = list(shape2)
    len_1 = len(shape1)
    len_2 = len(shape2)
    max_len = np.max([len_1, len_2])
    min_len = np.min([len_1, len_2])
    new_shape = [1]*max_len

    if len_1>len_2:
        new_shape = shape1
    elif len_2>len_1:
        new_shape = shape2

    for i in range(-1, -1-min_len, -1):
        if shape1[i] == shape2[i]:
            new_shape[i] = shape1[i]
        elif shape1[i] == 1:
            new_shape[i] = shape2[i]
        elif shape2[i] == 1:
            new_shape[i] = shape1[i]
        else:
            raise ValueError("Dimension Mismatch when trying to combine two data object. The first object has on axis {} a dimension of {}, where the second one has a dimension of {}".format(i, shape1[i], shape2[i]))
    return tuple(new_shape)

def upconvert_dimension(arr, shape):
    """
    upconverts the dimension of the array.

    Args:
        arr (np.ndarray) : input array
        shape (tuple) : wanted final shape.

    Returns:
        new_arr (np.ndarray) : upconverted array
    """

    if arr.shape == shape:
        return arr

    new_arr = np.empty(shape, arr.dtype)
    input_shape = arr.shape
    axis_filled = []
    for i in range(len(input_shape)):
        if input_shape[i] != 1:
            axis_filled.append(len(arr.shape) -1 - i)

    # swap axis to all data in lower axis numbers
    j = 0
    axis_swapped = []
    for i in axis_filled:
        if i!=j:
            arr = np.swapaxes(arr, len(arr.shape) -1 - i,len(arr.shape) -1 -  j)
            new_arr = np.swapaxes(new_arr, len(new_arr.shape) -1 - j, len(new_arr.shape) -1 - i)
            axis_swapped.append((len(arr.shape) -1 - i,len(arr.shape) -1 -  j, len(new_arr.shape) -1 - j, len(new_arr.shape) -1 - i))

        j += 1

    # reshape to easy copy dat form arr in new arr
    old_shape = new_arr.shape
    new_arr = new_arr.reshape([int(new_arr.size/arr.size),int(arr.size)])
    new_arr[:] = arr.flatten()

    new_arr = new_arr.reshape(old_shape)
    # swap back all the axis.
    for i in axis_swapped[::-1]:
        arr = np.swapaxes(arr, i[0], i[1])
        new_arr = np.swapaxes(new_arr, i[2], i[3])

    return new_arr


def reduce_arr(arr):
    """
    Return which elements on which axis are unique

    Args:
        arr (np.ndarray) : input array which to reduce to unique value

    Returns:
        reduced array(np.ndarray) : array with reduced data.
        data_axis (list) : the axises that have changing data.
    """
    ndim = len(arr.shape)
    data_axis = []
    slice_array = ()
    for i in range(ndim):
        slc_arr = i*(0,) + (slice(None),) + (ndim - i - 1) * (0,)
        new_arr = arr[slc_arr]
        if not np.all(new_arr == new_arr[0]):
            data_axis.append(ndim - i - 1)
            slice_array += (slice(None),)
        else:
            slice_array += (0,)
    red_ar = arr[slice_array]
    return red_ar, data_axis
