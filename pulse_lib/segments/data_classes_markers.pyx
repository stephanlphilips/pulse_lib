# distutils: language=c++

from libcpp.vector cimport vector
from libcpp.pair cimport pair
from libcpp cimport bool
from cython.operator import dereference as deref, postincrement as pp
import copy

import numpy as np


"""
temp copy of code of pulse data as those classes are not yet written in cython ... .
atm not possible to make extension class (stuct vs dict type)
"""

cdef class marker_data():
	cdef vector[pair[double, double]] my_marker_data
	cdef double start_time
	cdef double end_time
	cdef dict waveform_cache

	def __init__(self):
		self.start_time = 0
		self.end_time = 0
		self.waveform_cache = dict()

	def add_marker(self, double start, double stop):
		self.my_marker_data.push_back( (start + self.start_time, stop + self.start_time) )

		if stop + self.start_time > self.end_time:
			self.end_time = self.start_time + stop
	
	def reset_time(self):
		self.start_time = self.end_time

	@property
	def total_time(self):
		return self.end_time

	def slice_time(self, double start, double end):
		cdef vector[pair[double, double]].iterator write_iterator = self.my_marker_data.begin()
		cdef vector[pair[double, double]].iterator read_iterator = self.my_marker_data.begin()

		cdef bool in_range

		while(read_iterator != self.my_marker_data.end()):
			in_range = slice_out_marker_single(start, end, &deref(read_iterator))
			if in_range == True:
				# move not supported at the moment of writing, so not super effient, but acceptable as amount of data is small.
				write_iterator[0] = read_iterator[0]
				pp(write_iterator)

			pp(read_iterator)

		self.my_marker_data.erase(write_iterator, self.my_marker_data.end())
		self.end_time = end - start

	def get_vmin(self,sample_rate = 1e9):
		return 0

	def get_vmax(self,sample_rate = 1e9):
		return 1

	def integrate_waveform(self, pre_delay, post_delay, sample_rate):
		return 0

	def append(self, marker_data other, time):
		'''
		Append two segments to each other, where the other segment is places after the first segment. Time is the total time of the first segment.
		Args:
			other (marker_data) : other pulse data object to be appended
			time (double/None) : length that the first segment should be.

		** what to do with start time argument?
		'''
		cdef double c_time = self.end_time
		if time is not None:
			c_time = time
			self.slice_time(0, c_time)

		cdef marker_data other_shifted = other._shift_all_time(c_time)
		self.my_marker_data.insert(self.my_marker_data.end(), other_shifted.my_marker_data.begin(), other_shifted.my_marker_data.end())
		self.end_time += other.end_time

	def __copy__(self):
		cdef marker_data my_copy = marker_data()
		my_copy.my_marker_data = copy.copy(self.my_marker_data)
		my_copy.start_time = copy.copy(self.start_time)
		my_copy.end_time = copy.copy(self.end_time)


		return my_copy

	def _shift_all_time(self, double time_shift):
		'''
		Make a copy of all the data and shift all the time

		Args:
			time_shift (double) : shift the time
		Returns:
			data_copy_shifted (pulse_data) : copy of own data
		'''
		if time_shift <0 :
			raise ValueError("when shifting time, you cannot make negative times. Apply a positive shift.")
		
		cdef marker_data data_copy_shifted = copy.copy(self)
		cdef vector[pair[double, double]].iterator read_iterator = data_copy_shifted.my_marker_data.begin()

		while(read_iterator != data_copy_shifted.my_marker_data.end()):
			deref(read_iterator).first =  deref(read_iterator).first + time_shift
			deref(read_iterator).second =  deref(read_iterator).second + time_shift
			pp(read_iterator)

		return data_copy_shifted
		
	def __add(self, marker_data other):
		"""
		need to make a compiled function here, special python methods cannot access cython functions.
		"""
		return self.__add_cython__(other)

	def __add__(self, marker_data other):
		return self.__add(other)

	cdef marker_data __add_cython__(self, marker_data other):
		cdef marker_data new_data
		new_data = marker_data()
		new_data.my_marker_data.insert(new_data.my_marker_data.end(), self.my_marker_data.begin(), self.my_marker_data.end())

		new_data.my_marker_data.insert(new_data.my_marker_data.end(), other.my_marker_data.begin(), other.my_marker_data.end())
		new_data.start_time = self.start_time
		new_data.end_time= self.end_time

		if other.end_time > self.end_time:
			new_data.end_time = other.end_time

		return new_data

	def print_all(self):
		cdef vector[pair[double, double]].iterator read_iterator = self.my_marker_data.begin()

		while(read_iterator != self.my_marker_data.end()):
			print( deref(read_iterator))
			pp(read_iterator)

	def render(self, pre_delay = 0, post_delay = 0, sample_rate=1e9, clear_cache_on_exit = False):
		'''
		renders pulse
		Args:
			pre_delay (double) : amount of time to put before the sequence the rendering needs to start
			post_delay (double) : to which point in time the rendering needs to go
			sample_rate (double) : rate at which the AWG will be run
			clear_cache_on_exit (bool) : clear the cache on exit (e.g. when you uploaded this waveform to the AWG, remove reference so the garbarge collector can remove it). The ensured low memory occupation.
		returns
			pulse (np.ndarray) : numpy array of the pulse
		'''

		# If no render performed, generate full waveform, we will cut out the right size if needed

		if len(self.waveform_cache) == 0 or self.waveform_cache['sample_rate'] != sample_rate:
			pre_delay_wvf = pre_delay
			if pre_delay > 0:
				pre_delay_wvf = 0
			post_delay_wvf = post_delay
			if post_delay < 0:
				pre_delay_wvf = 0

			self.waveform_cache = {
				'sample_rate' : sample_rate,
				'waveform' : self.__render(sample_rate, pre_delay_wvf, post_delay_wvf),
				'pre_delay': pre_delay,
				'post_delay' : post_delay
			}

		# get the waveform
		my_waveform = self.get_resized_waveform(pre_delay, post_delay)
		
		# clear cache if needed
		if clear_cache_on_exit == True:
			self.waveform_cache = dict()

		return my_waveform

	def __render(self, sample_rate, pre_delay = 0.0, post_delay = 0.0):
		'''
		make a full rendering of the waveform at a predermined sample rate.
		'''
		# express in Gs/s
		sample_rate = sample_rate*1e-9
		sample_time_step = 1/sample_rate
		
		t_tot = self.total_time

		# get number of points that need to be rendered
		t_tot_pt = get_effective_point_number(t_tot, sample_time_step) + 1
		pre_delay_pt = - get_effective_point_number(pre_delay, sample_time_step)
		post_delay_pt = get_effective_point_number(post_delay, sample_time_step)

		my_sequence = np.zeros([int(t_tot_pt + pre_delay_pt + post_delay_pt)])

		cdef vector[pair[double, double]].iterator read_iterator = self.my_marker_data.begin()

		cdef int start
		cdef int stop
		while(read_iterator != self.my_marker_data.end()):
			start = get_effective_point_number(deref(read_iterator).first, sample_time_step) + pre_delay_pt
			stop = get_effective_point_number(deref(read_iterator).second, sample_time_step) + pre_delay_pt

			my_sequence[start:stop] = 1
			pp(read_iterator)

		return my_sequence

	def get_resized_waveform(self, pre_delay, post_delay):
		'''
		extend/shrink an existing waveform
		Args:
			pre_delay (double) : ns to add before
			post_delay (double) : ns to add after the waveform
		Returns:
			waveform (np.ndarray[ndim=1, dtype=double])
		'''

		sample_rate = self.waveform_cache['sample_rate']*1e-9
		sample_time_step = 1/sample_rate

		pre_delay_pt = get_effective_point_number(pre_delay, sample_time_step)
		post_delay_pt = get_effective_point_number(post_delay, sample_time_step)

		wvf_pre_delay_pt = get_effective_point_number(self.waveform_cache['pre_delay'], sample_time_step)
		wvf_post_delay_pt = get_effective_point_number(self.waveform_cache['post_delay'], sample_time_step)

		# points to add/remove from existing waveform
		n_pt_before = - pre_delay_pt + wvf_pre_delay_pt
		n_pt_after = post_delay_pt - wvf_post_delay_pt

		# if cutting is possible (prefered since no copying is involved)
		if n_pt_before <= 0 and n_pt_after <= 0:
			if n_pt_after == 0:
				return self.waveform_cache['waveform'][-n_pt_before:]
			else:
				return self.waveform_cache['waveform'][-n_pt_before: n_pt_after]
		else:
			n_pt = len(self.waveform_cache['waveform']) + n_pt_after + n_pt_before
			new_waveform =  np.zeros((n_pt, ))

			if n_pt_before > 0:
				new_waveform[0:n_pt_before] = self.my_pulse_data[0,1]
				if n_pt_after < 0:
					new_waveform[n_pt_before:] = self.waveform_cache['waveform'][:n_pt_after]
				elif n_pt_after == 0:
					new_waveform[n_pt_before:] = self.waveform_cache['waveform']
				else:
					new_waveform[n_pt_before:-n_pt_after] = self.waveform_cache['waveform']
			else:
				new_waveform[:-n_pt_after] = self.waveform_cache['waveform'][-n_pt_before:]

			if n_pt_after > 0:
				new_waveform[-n_pt_after:] =  self.my_pulse_data[-1,1]

			return new_waveform


cdef bool slice_out_marker_single(double start, double stop, pair[double, double] *start_stop_position):
	"""
	check if start stop falls in valid range.
	Return:
		True/False if start and stop are not in range

	Function also fixes the time in the pointer that is given.
	"""
	if start_stop_position.second <= start or start_stop_position.first >= stop:
		return False

	if start_stop_position.first < start:
		start_stop_position.first = start

	if start_stop_position.second > stop:
		start_stop_position.second = stop

	start_stop_position.first -= start
	start_stop_position.second -= start

	return True

cdef int get_effective_point_number(double time, double time_step):
	'''
	function that discretizes time depending on the sample rate of the AWG.
	Args:
		time (double): time in ns of which you want to know how many points the AWG needs to get there
		time_step (double) : time step of the AWG (ns)

	Returns:
		how many points you need to get to the desired time step.
	'''
	n_pt, mod = divmod(time, time_step)
	if mod > time_step/2:
		n_pt += 1

	return int(n_pt)
