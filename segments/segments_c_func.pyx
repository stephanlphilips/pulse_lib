# cython: profile=True


import numpy as np
cimport numpy as np
import cython


# from cpython cimport array
# from libc.stdlib import bsearch

# cdef int CustCmp( const void *a, const void *b ) with gil:    
# 	cdef double a_v = (< double*>a)[0]
# 	cdef double b_v = (< double*>b)[0]

# 	if a_v < b_v: return -1
# 	elif b_v < a_v: return 1
# 	else: return 0

# def interpolate_pulse(new_pulse, old_pulse):
# 	c_interpolate_pulse(new_pulse, old_pulse)


# cdef void c_interpolate_pulse(double[:,:] new_pulse, double[:,:] old_pulse):
# 	'''
# 	interpolate value in new_pulse that are not present in the old pulse array
# 	Args: 
# 		new_pulse: 2D array containing the timings and amplitudes desciribing the pulse
# 		old_pulse: 2D array, but from the old one
# 	Return:
# 		new_pulse, with interpolated values

# 	# note this only works if old pulse and new pulse do have the same starting argument.
# 	'''
# 	cdef bint skip_next_iteration = False
# 	cdef int len_new_pulse = new_pulse.shape[0]

# 	cdef double interpolation = 0

# 	cdef int *idx
# 	cdef int i, time_indice_a, time_indice_b, times, prev_point, next_point
# 	for i in range(len_new_pulse):
# 		times = 0

# 		if skip_next_iteration == True:
# 			skip_next_iteration = False
# 			continue

# 		# idx = np.searchsorted(old_pulse[:,0], new_pulse[i,0])

# 		idx = < int*>bsearch( new_pulse[i,0], old_pulse[:,0], 10, sizeof( double ), &CustCmp )

# 		if new_pulse[i,0] == old_pulse[idx,0]:
# 			times += 1
# 			time_indice_a = idx
# 			if  new_pulse[i +1,0] == old_pulse[idx,0]:
# 				times += 1
# 				time_indice_b = idx

# 		if times == 1:
# 			new_pulse[i,1] = old_pulse[time_indice_a,1]
# 			if i != len_new_pulse-1:
# 				if new_pulse[i+1,0] == new_pulse[i,0]:
# 					new_pulse[i+1,1] = old_pulse[time_indice_a,1]
# 					skip_next_iteration = True
# 		elif times == 2:
# 			new_pulse[i:i+1,:] = old_pulse[time_indice_a,:]
# 			new_pulse[i:i+2,:] = old_pulse[time_indice_b,:]

# 			skip_next_iteration = True
# 		elif new_pulse[i,0] >= old_pulse[-1,0]:
# 			new_pulse[i,1] = new_pulse[-1,1]
# 		else:
# 			prev_point = np.searchsorted(old_pulse[:,0],new_pulse[i,0]) - 1
# 			next_point = np.searchsorted(old_pulse[:,0],new_pulse[i,0], side='left')

# 			# print(next_point)
# 			# prev_point = np.where(old_pulse[:,0] <= new_pulse[i,0])[0][-1]
# 			# next_point = np.where(old_pulse[:,0] >= new_pulse[i,0])[0][ 0]
# 			# print("wanted", next_point)

# 			interpolation = calc_value_point_in_between(old_pulse[prev_point], old_pulse[next_point], new_pulse[i,0])
# 			new_pulse[i,1] = interpolation

# @cython.boundscheck(False)
def interpolate_pulses(np.ndarray[dtype=double, ndim=2] pulse_1, np.ndarray[dtype=double, ndim=2] pulse_2):
	return c_interpolate_pulses(pulse_1, pulse_2)

cdef c_interpolate_pulses(np.ndarray[dtype=double, ndim=2] pulse_1, np.ndarray[dtype=double, ndim=2] pulse_2):
	'''
	function that generates common times of two pulses. To be used to interpolate them
	Args:
		pulse_1 : np.ndarray (2d, double)
		pulse 2 : np.ndarray (2d, double)
	Returns:
		pulse_1_interpolated : np.ndarray (2d, double)
		pulse_2_interpolated : np.ndarray (2d, double)
	'''

	cdef double [:,:] pulse_1_view = pulse_1
	cdef double [:,:] pulse_2_view = pulse_2

	cdef np.ndarray new_times = np.empty(len(pulse_1_view)+ len(pulse_2_view), dtype=np.double)
	cdef np.ndarray new_data_1 = np.empty(len(pulse_1_view)+ len(pulse_2_view), dtype=np.double)
	cdef np.ndarray new_data_2 = np.empty(len(pulse_1_view)+ len(pulse_2_view), dtype=np.double)
	cdef np.ndarray missing_idx_1 = np.empty(len(pulse_2_view), dtype=np.int32)
	cdef np.ndarray missing_idx_2 = np.empty(len(pulse_1_view), dtype=np.int32)


	cdef double [:] new_times_view = new_times
	cdef double [:] new_data_1_view = new_data_1
	cdef double [:] new_data_2_view = new_data_2
	cdef int [:] missing_idx_1_view = missing_idx_1
	cdef int [:] missing_idx_2_view = missing_idx_2


	cdef int k = len(new_times_view)
	cdef int i = len(pulse_1_view) -1
	cdef int j = len(pulse_2_view) -1

	cdef int mi_1 = len(missing_idx_1_view) -1
	cdef int mi_2 = len(missing_idx_2_view) -1

	local_indice = k - 1

	while (i >= 0 or j >=0):
		k -= 1
		if j < 0  or (i>=0 and pulse_1_view[i,0] >= pulse_2_view[j,0]):
			if pulse_1_view[i,0] == pulse_2_view[j,0]:
				new_data_2_view[k] = pulse_2_view[j,1]
				j -=1
			else:
				missing_idx_2_view[mi_2] = k
				mi_2 -= 1

			new_times_view[local_indice] = pulse_1_view[i,0]
			new_data_1_view[k] = pulse_1_view[i,1]
			local_indice -= 1
			i -= 1
		else:
			new_times_view[local_indice] = pulse_2_view[j,0]
			local_indice -= 1
			new_data_2_view[k] = pulse_2_view[j,1]
			missing_idx_1_view[mi_1] = k
			mi_1 -= 1
			j -= 1

	pulse_1_interpolated = np.empty([len(new_times_view)-local_indice -1, 2], dtype=np.double)
	pulse_2_interpolated = np.empty([len(new_times_view)-local_indice -1, 2], dtype=np.double)
	cdef double [:,:] pulse_1_interpolated_view = pulse_1_interpolated
	cdef double [:,:] pulse_2_interpolated_view = pulse_2_interpolated


	pulse_1_interpolated_view[:,0] = new_times_view[local_indice+1:]
	pulse_1_interpolated_view[:,1] = new_data_1_view[local_indice+1:]

	pulse_2_interpolated_view[:,0] = new_times_view[local_indice+1:]
	pulse_2_interpolated_view[:,1] = new_data_2_view[local_indice+1:]

	cdef int my_const = -1 -mi_1

	for i in range(mi_1+1, len(missing_idx_1)):
		missing_idx_1_view[i] += my_const

	my_const = -1 -mi_1
	i = mi_2+1
	for i in range(mi_2+1, len(missing_idx_2)):
		missing_idx_2_view[i] += my_const

	add_missing_values(pulse_1_interpolated_view, missing_idx_1_view[mi_1+1:])
	add_missing_values(pulse_2_interpolated_view, missing_idx_2_view[mi_1+1:])
	
	return pulse_1_interpolated, pulse_2_interpolated

cdef void add_missing_values(double[:,:] pulse, int[:] indexes):
	'''
	Adds missing values to a pulse at the indexes given.
	Args:
		pulse (np.ndarray[dim=2])
		indexes ([:]) : index values that need to be filled in.
	Returns:
		None
	'''
	cdef int next_idx
	cdef int i = 0
	for i in range(len(indexes)):
		if indexes[i] == 0:
			pulse[0,1] = 0
			continue

		next_idx = get_next_value_idx(indexes, i)
		pulse[indexes[i], 1] = calc_value_point_in_between(pulse[indexes[i]-1], pulse[next_idx], pulse[indexes[i], 0])

cdef int get_next_value_idx(int[:] index, int i):
	cdef int idx = 0
	cdef bint value_found = False

	if i == len(index) -1 :
		return index[i]+1

	while(value_found != True):
		if i == len(index)-1:
			idx = index[i]+1
			value_found = True

		elif index[i] != index[i+1] -1 :
			idx = index[i]+1
			value_found = True
		i+=1

	return idx

def py_calc_value_point_in_between(ref1, ref2, time):
	return calc_value_point_in_between(ref1, ref2, time)

cdef double calc_value_point_in_between(double[:] ref1, double[:] ref2, double time):
		'''
		calculate amplitude of waveform at the certain time 
		Args:
			ref1 : time1 and amplitude1  (part of a pulse object)
			ref2 : time2 and amplitude2 at a different time (part of a pulse object)
			time : time at which you want to know to ampltitude (should be inbetween t1 and t2)
		Returns:
			amplitude at time 
		'''
		cdef double a = (ref2[1]- ref1[1])/(ref2[0]- ref1[0])
		cdef double c = ref1[1] - a*ref1[0]

		return a*time + c