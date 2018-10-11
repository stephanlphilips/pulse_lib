import numpy as np


def interpolate_pulses(pulse_1, pulse_2):
	'''
	function that generates common times of two pulses. To be used to interpolate them
	Args:
		pulse_1 : np.ndarray (2d, double)
		pulse 2 : np.ndarray (2d, double)
	Returns:
		pulse_1_interpolated : np.ndarray (2d, double)
		pulse_2_interpolated : np.ndarray (2d, double)
	'''

	data_1 = pulse_1[:,1]
	data_2 = pulse_2[:,1]
	time_series1 = pulse_1[:,0]
	time_series2 = pulse_2[:,0]
	
	new_times = [0]*(len(time_series1)+ len(time_series2))
	new_data_1 = [0]*(len(time_series1)+ len(time_series2))
	new_data_2 = [0]*(len(time_series1)+ len(time_series2))
	missing_idx_1 = [0]*(len(time_series2))
	missing_idx_2 = [0]*(len(time_series1))

	k = len(new_times)
	i = len(time_series1) -1
	j = len(time_series2) -1

	mi_1 = len(missing_idx_1) -1
	mi_2 = len(missing_idx_2) -1

	local_indice = k - 1

	while (i >= 0 or j >=0):
		k -= 1
		if j < 0  or (i>=0 and time_series1[i] >= time_series2[j]):
			if time_series1[i] == time_series2[j]:
				new_data_2[k] = data_2[j]
				j -=1
			else:
				missing_idx_2[mi_2] = k
				mi_2 -= 1

			new_times[local_indice] = time_series1[i]
			new_data_1[k] = data_1[i]
			local_indice -= 1
			i -= 1
		else:
			new_times[local_indice] = time_series2[j]
			local_indice -= 1
			new_data_2[k] = data_2[j]
			missing_idx_1[mi_1] = k
			mi_1 -= 1
			j -= 1

	pulse_1_interpolated = np.empty([len(new_times)-local_indice -1, 2], dtype = np.double)
	pulse_2_interpolated = np.empty([len(new_times)-local_indice -1, 2], dtype = np.double)

	pulse_1_interpolated[:,0] = new_times[local_indice+1:]
	pulse_1_interpolated[:,1] = new_data_1[local_indice+1:]

	pulse_2_interpolated[:,0] = new_times[local_indice+1:]
	pulse_2_interpolated[:,1] = new_data_2[local_indice+1:]

	add_missing_values(pulse_1_interpolated, np.array(missing_idx_1[mi_1+1:]) - mi_1 - 1)
	add_missing_values(pulse_2_interpolated, np.array(missing_idx_2[mi_1+1:]) - mi_2 - 1)
	
	return pulse_1_interpolated, pulse_2_interpolated


def add_missing_values(pulse, indexes):
	'''
	Adds missing values to a pulse at the indexes given.
	Args:
		pulse (np.ndarray[dim=2])
		indexes ([:]) : index values that need to be filled in.
	Returns:
		None
	'''
	print(pulse, indexes)
	for i in range(len(indexes)):
		if indexes[i] == 0:
			pulse[0,1] = 0
			continue

		next_idx = get_next_value_idx(pulse, indexes, i)
		print(next_idx)
		pulse[indexes[i], 1] = calc_value_point_in_between(pulse[indexes[i]-1], pulse[next_idx], pulse[indexes[i], 0])


def get_next_value_idx(pulse, index, i):
	idx = 0
	value_found = False
	if i == len(index) -1 :
		return index[i]+1

	while(value_found != True):
		if i == len(index)-1:
			idx = index[i]+1
			value_found = True

		elif index[i] != index[i+1] -1 :
			idx = index[i]+1
			print("luky:", idx)
			value_found = True
		i+=1

	return idx

def calc_value_point_in_between(ref1, ref2, time):
		'''
		calculate amplitude of waveform at the certain time 
		Args:
			ref1 : time1 and amplitude1  (part of a pulse object)
			ref2 : time2 and amplitude2 at a different time (part of a pulse object)
			time : time at which you want to know to ampltitude (should be inbetween t1 and t2)
		Returns:
			amplitude at time 
		'''
		a = (ref2[1]- ref1[1])/(ref2[0]- ref1[0])
		c = ref1[1] - a*ref1[0]

		return a*time + c


import time
from numpy import array
t1 =time.time()
a = np.asarray([array([0., 0.]), array([10.,  0.]), array([10., 50.]), array([200.,  50.]), array([200.,   0.]), array([250.,   0.])])
b = np.asarray([array([0., 0.]), array([60.,  0.]), array([60., 50.]), array([250.,  50.]), array([250.,   0.])]
)
t1 =time.time()
print(a)
p1, p2 = interpolate_pulses(a,b)
t2 = time.time()
print(p1,p2)
# print(t2-t1)