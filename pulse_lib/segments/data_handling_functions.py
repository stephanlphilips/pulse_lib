import numpy as np
import copy


def update_dimension(data, new_dimension_info):
	'''
	update dimension of the data object to the one specified in new dimension_info
	Args:
		data (np.ndarray[dtype = object]) : numpy object that contains all the segment data of every iteration.
		new_dimension_info (list/np.ndarray) : list of the new dimensions of the array

	Returns:
		data (np.ndarray[dtype = object]) : same as input data, but with new_dimension_info.
	'''

	new_dimension_info = np.array(new_dimension_info)

	for i in range(len(new_dimension_info)):
		if data.ndim < i+1:
			data = _add_dimensions(data, new_dimension_info[-i-1:])

		elif list(data.shape)[-i -1] != new_dimension_info[-i -1]:
			shape = list(data.shape)
			shape[-i-1] = new_dimension_info[-i-1]
			data = _extend_dimensions(data, shape)

	return data

def _add_dimensions(data, shape):
	"""
	Function that can be used to add and extra dimension of an array object. A seperate function is needed since we want to make a copy and not a reference.
	Note that only one dimension can be extended!
	Args:
		data (np.ndarray[dtype = object]) : numpy object that contains all the segment data of every iteration.
		shape (list/np.ndarray) : list of the new dimensions of the array
	"""
	new_data = np.empty(np.array(shape), dtype=object)
	for i in range(shape[0]):
		new_data[i] = cpy_numpy_shallow(data)
	return new_data

def _extend_dimensions(data, shape):
	'''
	Extends the dimensions of a existing array object. This is useful if one would have first defined sweep axis 2 without defining axis 1.
	In this case axis 1 is implicitly made, only harbouring 1 element.
	This function can be used to change the axis 1 to a different size.

	Args:
		data (np.ndarray[dtype = object]) : numpy object that contains all the segment data of every iteration.
		shape (list/np.ndarray) : list of the new dimensions of the array (should have the same lenght as the dimension of the data!)
	'''

	new_data = np.empty(np.array(shape), dtype=object)
	for i in range(len(shape)):
		if data.shape[i] != shape[i]:
			if i == 0:
				for j in range(len(new_data)):
					new_data[j] = cpy_numpy_shallow(data)
			else:
				new_data = new_data.swapaxes(i, 0)
				data = data.swapaxes(i, 0)

				for j in range(len(new_data)):
					new_data[j] = cpy_numpy_shallow(data)
				
				new_data = new_data.swapaxes(i, 0)


	return new_data

def cpy_numpy_shallow(data):
	'''
	Makes a shallow copy of an numpy object array.
	
	Args:
		data : data element
	'''
	if type(data) != np.ndarray:
		return copy(data)

	if data.shape == (1,):
		return copy.copy(data[0])

	shape = data.shape
	data_flat = data.flatten()
	new_arr = np.empty(data_flat.shape, dtype=object)

	for i in range(len(new_arr)):
		new_arr[i] = copy.copy(data_flat[i])

	return new_data

def cpy_numpy_shallow(data):
	'''
	Makes a shallow copy of an numpy object array.
	
	Args:
		data : data element
	'''
	if type(data) != np.ndarray:
		return copy(data)

	if data.shape == (1,):
		return copy.copy(data[0])

	shape = data.shape
	data_flat = data.flatten()
	new_arr = np.empty(data_flat.shape, dtype=object)

	for i in range(len(new_arr)):
		new_arr[i] = copy.copy(data_flat[i])

	new_arr = new_arr.reshape(shape)
	return new_arr

def loop_controller(func):
	'''
	Checks if there are there are parameters given that are loopable.

	If loop:
		* then check how many new loop parameters on which axis
		* extend data format to the right shape (simple python list used).
		* loop over the data and add called function

	if no loop, just apply func on all data (easy)
	'''
	def wrapper(*args, **kwargs):
		obj = args[0]

		loop_info_args = []
		loop_info_kwargs = []

		for i in range(1,len(args)):
			if type(args[i]) == linspace : 
				info = {
				'nth_arg': i,
				'name': args[i].name,
				'len': len(args[i]),
				'axis': args[i].axis,
				'data' : args[i].data}
				loop_info_args.append(info)

		for key in kwargs.keys():
			if type(kwargs[key]) == linspace : 
				info = {
				'nth_arg': key,
				'name': kwargs[key].name,
				'len': len(kwargs[key]),
				'axis': kwargs[key].axis,
				'data' : kwargs[key].data}
				loop_info_kwargs.append(info)
		
		
		for lp in loop_info_args:
			new_dim = get_new_dim_loop(obj.data.shape, lp)
			obj.data = update_dimension(obj.data, new_dim)
		
		for lp in loop_info_kwargs:
			new_dim = get_new_dim_loop(obj.data.shape, lp)
			obj.data = update_dimension(obj.data, new_dim)

		loop_over_data(func, obj.data, args, loop_info_args, kwargs, loop_info_kwargs)

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
			if arg['axis'] == n_dim-1:
				args_cpy[arg['nth_arg']] = args[arg['nth_arg']].data[i]
		for kwarg in kwargs_info:
			if arg['axis'] == n_dim-1:
				kwargs_cpy[kwargs_info['nth_arg']] = kwargs[kwargs_info['nth_arg']].data[i]

		if n_dim == 1:
			# we are at the lowest level of the loop.
			args_cpy[0].data_tmp = data[i]
			func(*args_cpy, **kwargs_cpy)
		else:
			# clean up args, kwards
			loop_over_data(func, data[i], args_cpy, args_info, kwargs_cpy, kwargs_info)

def get_new_dim_loop(current_dim, loop_spec):
	'''
	function to get new dimensions from a loop spec.
	
	Args:
		current_dim [tuple/array] : current dimensions of the data object 
		loop_spec [dict] : format of the loop
	
	Returns:
		new_dim [array] : new dimensions of the data obeject when one would include the loop spec
	'''
	current_dim = list(current_dim)
	new_dim = []
	if loop_spec["axis"] == -1:
		new_dim = [loop_spec["len"]] + current_dim
		# assign new axis.
		loop_spec["axis"] = len(new_dim) - 1
	else:
		if loop_spec["axis"] >= len(current_dim):
			new_dim = [1]*(loop_spec["axis"]+1)
			for i in range(len(current_dim)):
				new_dim[loop_spec["axis"]-len(current_dim)+1 + i] = current_dim[i]
			new_dim[0] = loop_spec["len"]
		else:
			if current_dim[-1-loop_spec["axis"]] == loop_spec["len"]:
				new_dim = current_dim
			elif current_dim[-1-loop_spec["axis"]] == 1:
				new_dim = current_dim
				new_dim[-1-loop_spec["axis"]] = loop_spec['len']
			else:
				raise ValueError("Dimensions on loop axis {} not compatible with previous loops\n\
					(current dimensions is {}, wanted is {}).\n\
					Please change loop axis or update the length.".format(loop_spec["axis"],
					current_dim[loop_spec["axis"]], loop_spec["len"]))

	return new_dim

def update_labels(data_object, loop_spec):
	pass

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


class linspace():
	"""docstring for linspace"""
	def __init__(self, start, stop, n_steps = 50, name = "undefined", unit = 'a.u.', axis = -1):
		self.data = np.linspace(start, stop, n_steps)
		self.name = name
		self.unit = unit
		self.axis = axis
	def __len__(self):
		return len(self.data)

