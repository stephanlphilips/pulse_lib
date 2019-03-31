"""
Class that is used to make DC pulses.
"""

import numpy as np
import datetime

from pulse_lib.segments.segment_base import last_edited, segment_base
from pulse_lib.segments.utility.data_handling_functions import loop_controller
from pulse_lib.segments.data_classes.data_pulse import pulse_data
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

	def add_reference_channel(self, channel_name, segment_data, multiplication_factor):
		'''
		Add channel reference, this can be done to make by making a pointer to another segment.
		Args:
			Channels_name (str): human readable name of the virtual gate.
			segment_data (segment_single): pointer so the segment corresponsing the the channel name
			multiplication_factor (float64): times how much this segment should be added to the current one.
		'''
		virtual_segment = {'name': channel_name, 'segment': segment_data, 'multiplication_factor': multiplication_factor}
		self.reference_channels.append(virtual_segment)

	def add_IQ_ref_channel(self, channel_name, pointer_to_channel, I_or_Q_part):
		'''
		Add a reference to an IQ channel. Same principle as for the virtual one.
		Args:
			channel_name (str): human readable name of the virtual channel
			pointer_to_channel (*segment_single_IQ): pointer to segment_single_IQ object
			I_or_Q_part (str) : 'I' or 'Q' to indicate that the reference is to the I or Q part of the signal.
		'''
		virtual_IQ_segment = {'name': channel_name, 'segment': pointer_to_channel, 'I/Q': I_or_Q_part}
		self.IQ_ref_channels.append(virtual_IQ_segment)
	
	@last_edited
	@loop_controller
	def reset_time(self, time=None, extend_only = False):
		'''
		resets the time back to zero after a certain point
		Args: 
			time (double) : (optional), after time to reset back to 0. Note that this is absolute time and not rescaled time.
		'''
		self.data_tmp.reset_time(time, extend_only)

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
		The pulse will have a relative phase (as this is needed to all IQ work).
		Args:
			start (double) : start time in ns of the pulse
			stop (double) : stop time in ns of the pulse
			amp (double) : amplitude of the pulse
			freq (double) : frequency of the pulse
			phase_offset (double) : offset in phase is needed
		'''
		self.data_tmp.add_sin_data(
			{
			'type' : 'std', 
			'start_time' : start + self.data_tmp.start_time,
			'stop_time' : stop + self.data_tmp.start_time,
			'amplitude' : amp,
			'frequency' : freq,
			'phase' : phase_offset})

	@last_edited
	@loop_controller
	def repeat(self, number):
		'''
		repeat a waveform n times.
		Args:
			number (int) : number of ties to repeat the waveform
		'''
		
		data_copy = copy.copy(self.data_tmp)
		for i in range(number-1):
			self.data_tmp.append(data_copy)

		# old code -- needs to be removed if the above function is validated.
		# this function can also be moved to segments base as it is generic.
		
		# if number <= 1:
		# 	return

		# # if repeating elemenets with double points in the start/end, we don't want them in the segement, so we will strip the first and add them later (back in the big sequence).
		# my_pulse_data_copy = copy.copy(self.data_tmp.my_pulse_data)
		# if my_pulse_data_copy[-1,0] < self.data_tmp.total_time:
		# 	my_pulse_data_copy = np.append(my_pulse_data_copy, [[self.data_tmp.total_time, my_pulse_data_copy[-1,1]]], axis=0)

		# front_pulse_corr = None
		# back_pulse_corr = None
		# # check if there is twice the same starting number 
		
		# if my_pulse_data_copy[0,0] == my_pulse_data_copy[1,0]:
		# 	front_pulse_corr = my_pulse_data_copy[0]
		# 	my_pulse_data_copy = my_pulse_data_copy[1:]

		# if my_pulse_data_copy[-1,0] == my_pulse_data_copy[-2,0]:
		# 	back_pulse_corr = my_pulse_data_copy[-1]
		# 	my_pulse_data_copy = my_pulse_data_copy[:-1]


		# pulse_data = np.zeros([my_pulse_data_copy.shape[0]*number, my_pulse_data_copy.shape[1]])

		# sin_data = []
		# total_time = self.data_tmp.total_time
		# indices = 0

		# for i in range(number):
		# 	new_pulse = copy.copy(my_pulse_data_copy)
			
		# 	new_pulse[:,0] +=  total_time*i
		# 	pulse_data[indices:indices + new_pulse.shape[0]] = new_pulse
		# 	indices += new_pulse.shape[0]

		# 	for sin_data_item in self.data_tmp.sin_data:
		# 		sin_data_item_new = copy.copy(sin_data_item)
		# 		sin_data_item_new['start_time'] += total_time*i
		# 		sin_data_item_new['stop_time'] += total_time*i
		# 		sin_data.append(sin_data_item_new)

		# if front_pulse_corr is not None:
		# 	corr_pulse = np.zeros([pulse_data.shape[0] + 1, pulse_data.shape[1]])
		# 	corr_pulse[1:] = pulse_data
		# 	corr_pulse[0] = front_pulse_corr
		# 	pulse_data = corr_pulse

		# if back_pulse_corr is not None:
		# 	corr_pulse = np.zeros([pulse_data.shape[0] + 1, pulse_data.shape[1]])
		# 	corr_pulse[:-1] = pulse_data
		# 	back_pulse_corr[0] = pulse_data[-1,0]
		# 	corr_pulse[-1] = back_pulse_corr
		# 	pulse_data = corr_pulse

		# self.data_tmp.my_pulse_data = pulse_data
		# self.data_tmp.sin_data = sin_data

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