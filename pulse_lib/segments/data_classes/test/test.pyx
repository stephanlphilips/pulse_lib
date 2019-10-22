# distutils: language = c++

cimport cython
from cpython cimport bool

import numpy as np
cimport numpy as np
from libcpp.vector cimport vector
from cython.operator import postincrement, dereference

import numbers
import copy

# note on linux is is long ... 
ctypedef long long longlong

cdef struct s_pulse_info:
	double start
	double stop
	
	double v_start
	double v_stop

	longlong index_start
	longlong index_stop

ctypedef s_pulse_info pulse_info

cdef class base_pulse_element:
	cdef pulse_info my_pulse_info

	def __init__(self, start, stop, v_start, v_stop):
		self.my_pulse_info.start = start
		self.my_pulse_info.stop = stop
		self.my_pulse_info.v_start = v_start
		self.my_pulse_info.v_stop = v_stop


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef class pulse_data_single_sequence():
	cdef public double _total_time
	cdef public vector[pulse_info] localdata
	cdef public bool re_render
	cdef double[:] voltage_data
	cdef double[:] time_data
	def __init__(self):
		# add empty pulse around 0 to make sure the pulse starts at that point
		pulse_init = base_pulse_element(0,3e-6,0,0)
		self.localdata = vector[pulse_info]()
		self.localdata.push_back(pulse_init.my_pulse_info)
		self._total_time = 0
		self.re_render = True
		self.voltage_data = np.empty([0])
		self.time_data = np.empty([0])

	def add_pulse(self, base_pulse_element pulse):
		self.localdata.push_back(pulse.my_pulse_info)
		
		if self._total_time < pulse.my_pulse_info.stop:
			self._total_time = pulse.my_pulse_info.stop

		self.re_render = True
	
	def append(self, other):
		"""
		append other pulse object to this one
		"""
		time_shift = self._total_time

		other.shift_time(time_shift)
		data = self.__add__(other)
		other.shift_time(-time_shift)

		self.localdata =  data.localdata
		self.re_render = True
		self._total_time = data._total_time

	def repeat(self, n):
		"""
		repeat n times the current segment.
		Args : 
			n (int) : number of times to repeat the current segment.
		"""

		cdef pulse_data_single_sequence pulse_data = copy.copy(self)
		cdef double total_time = self.total_time
		
		for i in range(n):
			pulse_data.shift_time(total_time)
			self._add_pulse(pulse_data)
			self._total_time += total_time


	def shift_time(self, double time):

		cdef vector[pulse_info].iterator it_localdata = self.localdata.begin()
		while(it_localdata != self.localdata.end()):
			dereference(it_localdata).start = dereference(it_localdata).start + time

			if dereference(it_localdata).stop != -1.:
				dereference(it_localdata).stop = dereference(it_localdata).stop + time
			
			postincrement(it_localdata)

		self._total_time += time
		self.re_render = True

	def __add__(self, other):
		cdef pulse_data_single_sequence pulse_data = copy.copy(self)

		if isinstance(other, pulse_data_single_sequence):
			pulse_data._add_pulse(other)

		elif isinstance(other, numbers.Number):
			pulse_data.localdata.push_back(base_pulse_element(0,-1., other, other).my_pulse_info)
		else:
			raise ValueError("adding up segment failed, data dype not recognize ({})".format(type(other)))

		if other._total_time > self._total_time:
			pulse_data._total_time = other._total_time

		pulse_data.re_render = True

		return pulse_data

	cdef _add_pulse(self, pulse_data_single_sequence other):

		for i in other.localdata:
			self.localdata.push_back(i)

	def __sub__(self, other):
		return self + other*(-1)

	def __mul__(self, other):
		cdef pulse_data_single_sequence pulse_data = copy.copy(self)
		if isinstance(other,numbers.Number):
			pulse_data._muliply(other)
		else:
			raise ValueError("adding up segment failed, data dype not recognize ({})".format(type(other)))

		return pulse_data

	cdef _muliply(self, double other):
		cdef vector[pulse_info].iterator it_localdata

		it_localdata = self.localdata.begin()
		while(it_localdata != self.localdata.end()):
			dereference(it_localdata).v_start = dereference(it_localdata).v_start*other
			dereference(it_localdata).v_stop = dereference(it_localdata).v_stop*other
			postincrement(it_localdata)

		self.re_render = True

	cdef __local_render(self):
		# relatively opimised..
		# tested for 1M pulses elements
		# --> ~156ms needed for full rendering on I9 cpu (note using c++ vectors gave significant speedup compared to python list due to typed index in localdata[i].start_index..)
		# speed currently limeted by sorting the times.

		cdef double[:] time_steps
		cdef double[:] time_steps_np
		cdef longlong[:] index_inverse
		cdef int j
		time_steps = np.empty([self.localdata.size()*4], dtype = np.double)
		# if putting too low, sometimes not nough 
		cdef double t_offset = 1e-6
		# not typed, this might slowdown, but performance is good enough atm.

		j = 0
		# 7ms
		cdef vector[pulse_info].iterator it_localdata = self.localdata.begin()

		while(it_localdata != self.localdata.end()):
			time_steps[j] = (dereference(it_localdata).start)
			time_steps[j + 1] = (dereference(it_localdata).start+t_offset)
			if dereference(it_localdata).stop == -1.:
				time_steps[j + 2] = self._total_time-t_offset
				time_steps[j + 3] = self._total_time
			else:
				time_steps[j + 2] = (dereference(it_localdata).stop-t_offset)
				time_steps[j + 3] = (dereference(it_localdata).stop)
			j+=4

			postincrement(it_localdata)


		# 130ms
		time_steps_np, index_inverse = np.unique(time_steps, return_inverse=True)

		j = 0
		
		# 5 ms
		it_localdata = self.localdata.begin()

		while(it_localdata != self.localdata.end()):
			dereference(it_localdata).index_start = index_inverse[j+1]
			dereference(it_localdata).index_stop = index_inverse[j+2]
			j += 4

			postincrement(it_localdata)


		cdef double[:] voltage_data = np.zeros([len(time_steps_np)])
		cdef double[:] time_step
		cdef double delta_v
		cdef double min_time
		cdef double max_time 
		cdef double rescaler

		# 20 ms
		it_localdata = self.localdata.begin()

		while(it_localdata != self.localdata.end()):
			delta_v = dereference(it_localdata).v_stop-dereference(it_localdata).v_start
			min_time = time_steps_np[dereference(it_localdata).index_start]
			max_time = time_steps_np[dereference(it_localdata).index_stop]

			rescaler = delta_v/(max_time-min_time)
			
			for j in range(dereference(it_localdata).index_start, dereference(it_localdata).index_stop+1):
				voltage_data[j] += dereference(it_localdata).v_start + (time_steps_np[j] - min_time)*rescaler

			postincrement(it_localdata)


		# # clean up data (few ms)
		cdef double[:] new_data_time = np.empty([len(voltage_data)])
		cdef double[:] new_data_voltage = np.empty([len(voltage_data)])

		new_data_time[0] = time_steps_np[0]
		new_data_voltage[0] = voltage_data[0]

		cdef int len_time_steps = len(time_steps_np)
		cdef double corr_offset = 2*t_offset #needs to be a bit bigger to correct for numberical errors.
		j = 1
		cdef int k = 1

		while( j < len_time_steps-1):
			if time_steps_np[j+1] - time_steps_np[j] < corr_offset and time_steps_np[j] - time_steps_np[j-1] < corr_offset:
				j+=1

			new_data_time[k] = time_steps_np[j]
			new_data_voltage[k] = voltage_data[j]
			j+= 1
			k+= 1
		if j < len_time_steps:
			new_data_time[k] = time_steps_np[j] 
			new_data_voltage[k] = voltage_data[j]
			k+= 1

		return new_data_time[:k], new_data_voltage[:k]

	def slice_time(self, double start, double stop):
		cdef vector[pulse_info].iterator it_localdata

		self._total_time = 0.
		it_localdata = self.localdata.begin()
		while(it_localdata != self.localdata.end()):
			
			if dereference(it_localdata).start < start:
				dereference(it_localdata).start = start
			if dereference(it_localdata).stop > stop:
				dereference(it_localdata).stop = stop

			if dereference(it_localdata).start < dereference(it_localdata).stop:
				dereference(it_localdata).start = dereference(it_localdata).start - start
				dereference(it_localdata).stop = dereference(it_localdata).stop - start
				self._total_time = dereference(it_localdata).stop
				postincrement(it_localdata)

			else:
				it_localdata = self.localdata.erase(it_localdata)

	def __copy__(self):
		cpy = pulse_data_single_sequence()
		cpy._total_time = self._total_time
		cpy.localdata = self.localdata

		return cpy

	@property
	def pulse_data(self):
		if self.re_render == True:
			self.time_data, self.voltage_data = self.__local_render()
		return (self.time_data, self.voltage_data)

	@property
	def total_time(self):
		return self._total_time

	@property
	def v_max(self):
		return np.max(self.pulse_data[1])

	@property
	def v_min(self):
		return np.min(self.pulse_data[1])