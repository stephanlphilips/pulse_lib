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
	print(new_dimension_info)
	for i in range(len(new_dimension_info)):
		print(data.ndim, data.shape)
		if data.ndim < i+1:
			print("dimensions extended")
			data = _add_dimensions(data, new_dimension_info[-i+1:])
			print(data.shape)
		if list(data.shape)[-i -1] != new_dimension_info[-i -1]:
			shape = list(data.shape)
			shape[-i-1] = new_dimension_info[-i-1]
			data = _extend_dimensions(data, shape)

	return data

def _add_dimensions(data, shape):
	"""
	function that can be used to add and extra dimension of an array object. A seperate function is needed since we want to make a copy and not a reference.
	Note that only one dimension can be extended!
	Args:
		data (np.ndarray[dtype = object]) : numpy object that contains all the segment data of every iteration.
		shape (list/np.ndarray) : list of the new dimensions of the array
	"""
	new_data = np.empty(np.array(shape), dtype=object)
	for i in range(shape[0]):
		new_data[i] = copy.deepcopy(data)
	return new_data

def _extend_dimensions(data, shape):
	'''
	extends the dimensions of a existing array object. This is useful if one would have first defined sweep axis 2 without defining axis 1.
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
					new_data[j] = copy.deepcopy(data)
			else:
				new_data = new_data.swapaxes(i, 0)
				data = data.swapaxes(i, 0)

				for j in range(len(new_data)):
					new_data[j] = copy.deepcopy(data)
				
				new_data = new_data.swapaxes(i, 0)


	return new_data
