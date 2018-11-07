import numpy as np


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
