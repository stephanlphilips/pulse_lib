import segments_c_func as seg_func

import numpy as np
import copy


class pulse_data():
	"""object that saves all the pulse data that is present in an segment object.
	This object support all the fundametal operations needed to define the segments."""
	def __init__(self):
		self.my_pulse_data = np.zeros([1,2], dtype=np.double)
		self.sin_data = []
		self.numpy_data = []

	def add_pulse_data(self, input):
		self.my_pulse_data = self._add_up_pulse_data(input)

	def add_sin_data(self, input):
		self.sin_data.append(input)

	def add_numpy_data(self, input):
		raise NotImplemented

	@property
	def total_time(self,):
		total_time = 0
		for sin_data_item in self.sin_data:
			if sin_data_item['stop_time'] > total_time:
				total_time = sin_data_item['stop_time']

		if self.my_pulse_data[-1,0] > total_time:
			total_time = self.my_pulse_data[-1,0]
		return total_time

	def get_vmax(self):
		'''
		calculate the maximum voltage in the current segment_single.

		Mote that the function now will only look at pulse data and sin data. It will count up the maxima of both the get to a absulute maxima
		this could be done more exacly by first rendering the whole sequence, and then searching for the maximum in there.
		Though current one is faster. When limiting, this should be changed (though here one shuold implment some smart chaching to avaid havinf to calculate the whole waveform twice).
		'''
		max_pulse_data = np.max(self.my_pulse_data[:,1])
		max_amp_sin = 0.0

		for i in self.sin_data:
			if max_amp_sin < i['amplitude']:
				max_amp_sin = i['amplitude']
		return max_pulse_data + max_amp_sin

	def get_vmin(self):
		'''
		calculate the maximum voltage in the current segment_single.

		Mote that the function now will only look at pulse data and sin data. It will count up the maxima of both the get to a absulute maxima
		this could be done more exacly by first rendering the whole sequence, and then searching for the maximum in there.
		Though current one is faster. When limiting, this should be changed (though here one shuold implment some smart chaching to avaid havinf to calculate the whole waveform twice).
		'''

		max_pulse_data = np.min(self.my_pulse_data[:,1])
		max_amp_sin = 0

		for i in self.sin_data:
			if max_amp_sin < i['amplitude']:
				max_amp_sin = i['amplitude']

		return max_pulse_data - max_amp_sin

	def _add_up_pulse_data(self, new_pulse):
		'''
		add a pulse up to the current pulse in the memory.
		new_pulse --> default format as in the add_pulse function
		'''
		my_pulse_data_copy = self.my_pulse_data
		# step 1: make sure both pulses have the same length
		if self.total_time < new_pulse[-1,0]:
			to_insert = [[new_pulse[-1,0],my_pulse_data_copy[-1,1]]]
			my_pulse_data_copy = self._insert_arrays(my_pulse_data_copy, to_insert, len(my_pulse_data_copy)-1)
		elif self.total_time > new_pulse[-1,0]:
			to_insert = [[my_pulse_data_copy[-1,0],new_pulse[-1,1]]]
			new_pulse = self._insert_arrays(new_pulse, to_insert, len(new_pulse)-1)
			
		my_pulse_data_tmp, new_pulse_tmp = seg_func.interpolate_pulses(my_pulse_data_copy, new_pulse)

		final_pulse = np.zeros([len(my_pulse_data_tmp),2])
		final_pulse[:,0] = my_pulse_data_tmp[:,0]
		final_pulse[:,1] +=  my_pulse_data_tmp[:,1]  + new_pulse_tmp[:,1]

		return final_pulse

	@staticmethod
	def _insert_arrays(src_array, to_insert, insert_position):
		'''
		insert pulse points in array
		Args:
			src_array : 2D pulse table
			to_insert : 2D pulse table to be inserted in the source
			insert_position: after which point the insertion needs to happen
		'''

		# calcute how long the piece is you want to insert
		dim_insert = len(to_insert)
		insert_position += 1

		new_arr = np.zeros([src_array.shape[0]+dim_insert, src_array.shape[1]])
		
		new_arr[:insert_position, :] = src_array[:insert_position, :]
		new_arr[insert_position:(insert_position + dim_insert), :] = to_insert
		new_arr[(insert_position + dim_insert):] = src_array[insert_position :]

		return new_arr

	def __copy__(self):
		my_copy = pulse_data()
		my_copy.my_pulse_data = copy.copy(self.my_pulse_data)
		my_copy.sin_data = copy.copy(self.sin_data)
		my_copy.numpy_data = copy.copy(self.numpy_data)
		return my_copy

class IQ_data():
	"""class that manages the data used for generating IQ data
	"""
	def __init__(self, LO):
		self.LO = LO
		self.simple_IQ_data = []
		self.MOD_IQ_data = []
		self.numpy_IQ_data = []

	def add_simple_data(self, input_dict):
		self.simple_IQ_data.append(input_dict)
	
	def add_mod_data (self, input_dict):
		self.simple_IQ_data.append(input_dict)

	def add_numpy_IQ(self, input_dict):
		self.numpy_IQ_data.append(input_dict)

	def __copy__(self,):
		my_copy = IQ_data(self.LO)
		my_copy.simple_IQ_data = copy.copy(self.simple_IQ_data)
		my_copy.MOD_IQ_data = copy.copy(self.MOD_IQ_data)
		my_copy.numpy_IQ_data = copy.copy(self.numpy_IQ_data)
		return my_copy

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

class linspace():
	"""docstring for linspace"""
	def __init__(self, start, stop, n_steps = 50, name = "undefined", unit = 'a.u.', axis = -1):
		self.data = np.linspace(start, stop, n_steps)
		self.name = name
		self.unit = unit
		self.axis = axis
	def __len__(self):
		return len(self.data)

