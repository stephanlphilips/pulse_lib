import numpy as np
# import pulse_lib.segments.data_classes_markers as mk

shape = (12,34, 3)



from dataclasses import dataclass

@dataclass
class reset_times_detailed():
	axis : list
	shape : list
	times : np.ndarray


times_1 = np.ones([5,1])
times_2 = np.ones([1,12])
times_3 = np.ones([1,1,20]) 

times_1[:,0] = np.linspace(1,5,5)
test = slice(0,len(times_1))
# print(times_1[test])
# print(times_1)


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

	for i in range(ndim):
		new_arr = arr
		if i != ndim -1:
			new_arr = np.swapaxes(new_arr, i, ndim-1)

		for j in range(ndim - 1):
			new_arr = new_arr[0]

		if len(new_arr) > 0 and np.array_equal(new_arr, np.full(new_arr.shape, new_arr[0])) == False:
			data_axis.append(i)

	data_axis = [len(arr.shape) - i - 1 for i in data_axis]

	j = 0
	for i in data_axis:
		if i != j:
			arr = np.swapaxes(arr, len(arr.shape) - i -1, len(arr.shape) - j -1)		
		j += 1

	for i in range(len(arr.shape) - len(data_axis)):
		arr = arr[0]
	return arr, data_axis

# j = 4
# t = np.linspace(0,10,4*3)
# # t = np.ones([j])

# # t2 = upconvert_dimension(t, (4,5,3))
# # print(t2)
# # a= (reduce_arr(t2))
# # print(a)
