import numpy as np
import copy


import segments_c_func as seg_func


class pulse_data():
	"""object that saves all the pulse data that is present in an segment object.
	This object support all the fundametal operations needed to define the segments."""
	def __init__(self):
		self.my_pulse_data = np.zeros([1,2], dtype=np.double)
		self.sin_data = []
		self.sim_mod_data = []

		self.numpy_data = []

		self.start_time = 0

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

	def reset_time(self,):
		self.start_time = self.total_time

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

	def render(self, start_time = 0, stoptime = 0, sample_rate=1e9):
		'''
		renders pulse
		Args:
			start_time (double) : from which points the rendering needs to start
			stop_time (double) : to which point the rendering needs to go (default (-1), to entire segement)
			sample_rate (double) : rate at which the AWG will be run

		returns
			pulse (np.ndarray) : numpy array of the pulse
		'''
		raise NotImplemented
		
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

	def __add__(self, other):
		'''
		define addition operator for segment_single
		'''
		new_data = pulse_data()
		if type(other) is pulse_data:
			if len(other.my_pulse_data) == 1:
				new_data.my_pulse_data = copy.copy(self.my_pulse_data)
			elif len(self.my_pulse_data) == 1:
				new_data.my_pulse_data = copy.copy(other.my_pulse_data)
			else:
				new_data.my_pulse_data = self._add_up_pulse_data(other.my_pulse_data)

			sin_data = copy.copy(self.sin_data)
			sin_data.extend(other.sin_data)
			new_data.sin_data = sin_data
		elif type(other) == int or type(other) == float:
			new_pulse = copy.copy(self.my_pulse_data)
			new_pulse[:,1] += other
			new_data.my_pulse_data = new_pulse
			new_data.sin_data = self.sin_data

		else:
			raise TypeError("Please add up segment_single type or a number ")

		return new_data

	def __mul__(self, other):
		'''
		muliplication operator for segment_single
		'''
		new_data = pulse_data()

		if type(other) is pulse_data:
			raise NotImplemented
		elif type(other) == int or type(other) == float or type(other) == np.float64:
			new_pulse = copy.copy(self.my_pulse_data)
			new_pulse[:,1] *= other
			new_data.my_pulse_data = new_pulse
			sin_data = []
			for i in self.sin_data:
				my_sin = copy.copy(i)
				my_sin['amplitude'] *=other
				sin_data.append(my_sin)

			new_data.sin_data = sin_data
		else:
			raise TypeError("muliplication shoulf be done with a number, type {} not supported".format(type(other)))
		
		return new_data

class IQ_data():
	"""
	class that manages the data used for generating IQ data
	"""
	def __init__(self, LO):
		self.LO = LO
		self.simple_IQ_data = []
		self.MOD_IQ_data = []
		self.numpy_IQ_data = []
		self.start_time = 0
		self.global_phase = 0

	def add_simple_data(self, input_dict):
		self.simple_IQ_data.append(input_dict)
	
	def add_mod_data (self, input_dict):
		self.simple_IQ_data.append(input_dict)

	def add_numpy_IQ(self, input_dict):
		self.numpy_IQ_data.append(input_dict)

	@property
	def total_time(self,):
		total_time = 0
		for IQ_data_item in self.simple_IQ_data:
			if IQ_data_item['stop_time'] > total_time:
				total_time = IQ_data_item['stop_time']

		for IQ_data_item in self.MOD_IQ_data:
			if IQ_data_item['stop_time'] > total_time:
				total_time = IQ_data_item['stop_time']

		for IQ_data_item in self.numpy_IQ_data:
			if IQ_data_item['stop_time'] > total_time:
				total_time = IQ_data_item['stop_time']

		return total_time

	def reset_time(self,):
		self.start_time = self.total_time

	def __copy__(self,):
		my_copy = IQ_data(self.LO)
		my_copy.simple_IQ_data = copy.copy(self.simple_IQ_data)
		my_copy.MOD_IQ_data = copy.copy(self.MOD_IQ_data)
		my_copy.numpy_IQ_data = copy.copy(self.numpy_IQ_data)
		my_copy.global_phase = copy.copy(self.global_phase)
		my_copy.start_time = copy.copy(self.start_time)
		return my_copy

	def get_IQ_data(self, I_or_Q):
		"""
		get data object containing the I or Q part of the IQ signal
		Args:
			I_or_Q (str) : string 'I' or 'Q' to indicate which part of the signal to return
		Returns:
			new_data (pulse_data) : normal pulse_data object.
		"""
		new_data = pulse_data()
		
		for i in self.simple_IQ_data:
			my_input = copy.copy(i)
			my_input['frequency'] -= self.LO
			if I_or_Q == 'Q':
				my_input['phase'] += np.pi/2
			new_data.add_sin_data(my_input)

		return new_data


	def __add__(self,):
		# aussume we do not need this.
		raise NotImplemented

	def __mul__(self):
		# aussume we do not need this.
		raise NotImplemented