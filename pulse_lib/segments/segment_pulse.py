"""
Class that is used to make DC pulses.
"""

import numpy as np
import datetime

from pulse_lib.segments.segment_base import last_edited, segment_base
from pulse_lib.segments.utility.data_handling_functions import loop_controller
from pulse_lib.segments.data_classes.data_pulse import pulse_data
from pulse_lib.segments.data_classes.data_IQ import IQ_data_single
import copy

class segment_pulse(segment_base):
	'''
	Class defining single segments for one sequence.
	'''
	def __init__(self, name, segment_type = 'render'):
		'''
		Args:
			name (str): name of the segment usually the channel name
			segment_type (str) : type of the segment (e.g. 'render' --> to be rendered, 'virtual'--> no not render.)
		'''
		super().__init__(name, pulse_data() ,segment_type)


	@last_edited
	@loop_controller
	def add_pulse(self,array):
		'''
		Add manually a pulse.
		Args: 
			array (list): array with times of the pulse.

		format array: 
		[[t0, Amp0],[t1, Amp1], ... [tn, Ampn]]
		'''
		if array[0] != [0.,0.]:
			array = [[0,0]] + array
		if self.data_tmp.start_time != 0:
			array = [[0,0]] + array

		arr = np.asarray(array)
		arr[1:,0] = arr[1:,0] + self.data_tmp.start_time 

		self.data_tmp.add_pulse_data(arr)

	@last_edited
	@loop_controller
	def add_block(self,start,stop, amplitude):
		'''
		add a block pulse on top of the existing pulse.
		'''
		if start != 0:
			pulse = np.array([[0,0], [start + self.data_tmp.start_time, 0], [start + self.data_tmp.start_time,amplitude], [stop + self.data_tmp.start_time, amplitude], [stop + self.data_tmp.start_time, 0]], dtype=np.double)
		else:
			pulse = np.array([[start + self.data_tmp.start_time, 0], [start + self.data_tmp.start_time,amplitude], [stop + self.data_tmp.start_time, amplitude], [stop + self.data_tmp.start_time, 0]], dtype=np.double)

		self.data_tmp.add_pulse_data(pulse)
	
	@last_edited
	@loop_controller
	def add_ramp(self, start, stop, amplitude, keep_amplitude=False):
		'''
		Makes a linear ramp
		Args:
			start (double) : starting time of the ramp
			stop (double) : stop time of the ramp
			amplitude : total hight of the ramp, starting from the base point
			keep_amplitude : when pulse is done, keep reached amplitude for time infinity
		'''
		if keep_amplitude == False:
			if start != 0:
				pulse = np.array([[0,0], [start + self.data_tmp.start_time, 0], [stop + self.data_tmp.start_time, amplitude], [stop + self.data_tmp.start_time, 0]], dtype=np.double)
			else:
				pulse = np.array([[start + self.data_tmp.start_time, 0],  [stop + self.data_tmp.start_time, amplitude], [stop + self.data_tmp.start_time, 0]], dtype=np.double)
		else:
			if start != 0:
				pulse = np.array([[0,0], [start + self.data_tmp.start_time, 0], [stop + self.data_tmp.start_time, amplitude] ], dtype=np.double)
			else:
				pulse = np.array([[start + self.data_tmp.start_time, 0],  [stop + self.data_tmp.start_time, amplitude] ], dtype=np.double)

		self.data_tmp.add_pulse_data(pulse)

	@last_edited
	@loop_controller
	def add_ramp_ss(self, start, stop, start_amplitude, stop_amplitude, keep_amplitude=False):
		'''
		Makes a linear ramp (with start and stop amplitude)
		Args:
			start (double) : starting time of the ramp
			stop (double) : stop time of the ramp
			amplitude : total hight of the ramp, starting from the base point
			keep_amplitude : when pulse is done, keep reached amplitude for time infinity
		'''
		if keep_amplitude == False:
			if start != 0:
				pulse = np.array([[0,0], [start + self.data_tmp.start_time, 0],[start + self.data_tmp.start_time, start_amplitude], 
					[stop + self.data_tmp.start_time, stop_amplitude], [stop + self.data_tmp.start_time, 0]], dtype=np.double)
			else:
				pulse = np.array([[start + self.data_tmp.start_time, 0],[start + self.data_tmp.start_time, start_amplitude],
					[stop + self.data_tmp.start_time, stop_amplitude], [stop + self.data_tmp.start_time, 0]], dtype=np.double)
		else:
			if start != 0:
				pulse = np.array([[0,0], [start + self.data_tmp.start_time, 0],[start + self.data_tmp.start_time, start_amplitude], [stop + self.data_tmp.start_time, stop_amplitude] ], dtype=np.double)
			else:
				pulse = np.array([[start + self.data_tmp.start_time, 0],[start + self.data_tmp.start_time, start_amplitude],  [stop + self.data_tmp.start_time, stop_amplitude] ], dtype=np.double)

		self.data_tmp.add_pulse_data(pulse)

	@last_edited
	@loop_controller
	def wait(self, wait):
		'''
		wait for x ns after the lastest wave element that was programmed.
		Args:
			wait (double) : time in ns to wait
		'''
		amp_0 = self.data_tmp.my_pulse_data[-1,1]
		t0 = self.data_tmp.total_time

		pulse = np.asarray([[t0, 0],[wait+t0, 0]], dtype=np.double)

		self.data_tmp.add_pulse_data(pulse)

	@last_edited
	@loop_controller
	def add_sin(self, start, stop, amp, freq, phase_offset=0):
		'''
		add a sinus to the current segment, parameters should be self exlenatory.
		The pulse will have a not have a relative phase phase.
		Args:
			start (double) : start time in ns of the pulse
			stop (double) : stop time in ns of the pulse
			amp (double) : amplitude of the pulse
			freq (double) : frequency of the pulse
			phase_offset (double) : offset in phase is needed
		'''
		self.data_tmp.add_MW_data(IQ_data_single(start, stop, amp, freq, phase_offset))

	@last_edited
	@loop_controller
	def add_np(self,start, array):
		raise NotImplemented



if __name__ == '__main__':
	import matplotlib.pyplot as plt

	s = segment_pulse("test")
	from pulse_lib.segments.utility.looping import linspace
	s.add_block(0, 10, linspace(2,10,50, unit=("ee")))
	# print(s.loops)
	# print(s.units)