"""
Class that is used to make DC pulses.
"""

import numpy as np

from pulse_lib.segments.segment_base import last_edited, segment_base
from pulse_lib.segments.utility.data_handling_functions import loop_controller
from pulse_lib.segments.data_classes.data_pulse import pulse_data
from pulse_lib.segments.data_classes.data_IQ import IQ_data_single
from pulse_lib.segments.utility.setpoint_mgr import setpoint_mgr
from pulse_lib.segments.data_classes.data_pulse_core import base_pulse_element
from pulse_lib.segments.segment_IQ import segment_IQ
from dataclasses import dataclass
import copy

@dataclass
class IQ_render_info:
	"""
	structure to save relevant information about the rendering of the IQ channels (for is in channel object).
	"""
	LO : float
	virtual_channel_name: str
	virtual_channel_pointer: segment_IQ #TODO fix to segment_IQ data type, needs to be post loaded somehow. 
	IQ_render_option : str
	image_render_option : str


class segment_pulse(segment_base):
	'''
	Class defining single segments for one sequence.
	'''
	def __init__(self, name, HVI_variable_data = None,segment_type = 'render'):
		'''
		Args:
			name (str): name of the segment usually the channel name
			HVI_variable_data (segment_HVI_variables) : segment used to keep variables that can be used in HVI.
			segment_type (str) : type of the segment (e.g. 'render' --> to be rendered, 'virtual'--> no not render.)
		'''
		super().__init__(name, pulse_data(), HVI_variable_data,segment_type)


	@last_edited
	@loop_controller
	def add_block(self,start,stop, amplitude):
		'''
		add a block pulse on top of the existing pulse.
		'''

		pulse = base_pulse_element(start + self.data_tmp.start_time,stop + self.data_tmp.start_time, amplitude, amplitude)
		self.data_tmp.add_pulse_data(pulse)
		return self.data_tmp
	
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
		
		if keep_amplitude == True:
			pulse = base_pulse_element(start + self.data_tmp.start_time,-1., 0., amplitude)
		else:
			pulse = base_pulse_element(start + self.data_tmp.start_time,stop + self.data_tmp.start_time, 0, amplitude)

		self.data_tmp.add_pulse_data(pulse)
		return self.data_tmp

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
		if keep_amplitude == True:
			pulse = base_pulse_element(start + self.data_tmp.start_time,-1, start_amplitude, stop_amplitude)
		else:
			pulse = base_pulse_element(start + self.data_tmp.start_time,stop + self.data_tmp.start_time, start_amplitude, stop_amplitude)

		self.data_tmp.add_pulse_data(pulse)
		return self.data_tmp

	@last_edited
	@loop_controller
	def wait(self, wait):
		'''
		wait for x ns after the lastest wave element that was programmed.
		Args:
			wait (double) : time in ns to wait
		'''
		t0 = self.data_tmp.total_time

		pulse = base_pulse_element(t0,wait+t0, 0, 0)
		self.data_tmp.add_pulse_data(pulse)
		return self.data_tmp

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
		self.data_tmp.add_MW_data(IQ_data_single(start + self.data_tmp.start_time, stop + self.data_tmp.start_time, amp, freq, phase_offset))
		return self.data_tmp

	@last_edited
	@loop_controller
	def add_np(self,start, array):
		raise NotImplemented

	def add_reference_channel(self, virtual_channel_reference_info):
		'''
		Add channel reference, this can be done to make by making a pointer to another segment.
		Args:
			virutal_channel_reference_info (dataclass): (defined in pulse_lib.virtual_channel_constructor)
				name (str): human readable name of the virtual gate.
				segment_data (segment_pulse): pointer so the segment corresponsing the the channel name
				multiplication_factor (float64): times how much this segment should be added to the current one.
		'''
		self.reference_channels.append(virtual_channel_reference_info)

	def add_IQ_channel(self, LO, channel_name, pointer_to_channel, I_or_Q_part, image):
		'''
		Add a reference to an IQ channel. Same principle as for the virtual one.
		Args:
			LO (float) : frequecy at which MW source runs (needed to calculate final IQ signal.)
			channel_name (str): human readable name of the virtual channel
			pointer_to_channel (*segment_single_IQ): pointer to segment_single_IQ object
			I_or_Q_part (str) : 'I' or 'Q' to indicate that the reference is to the I or Q part of the signal.
			image (str) : '+' / '-', take the image of the signal (needed for differential inputs)
		'''
		self.IQ_ref_channels.append(IQ_render_info(LO, channel_name, pointer_to_channel, I_or_Q_part, image))

	@last_edited
	@loop_controller
	def repeat(self, number):
		'''
		repeat a waveform n times.
		Args:
			number (int) : number of ties to repeat the waveform
		'''
		self.data_tmp.repeat(number)
		return self.data_tmp

	def __copy__(self):
		cpy = segment_pulse(self.name)
		return self._copy(cpy)
	



if __name__ == '__main__':
	import matplotlib.pyplot as plt

	from pulse_lib.segments.segment_HVI_variables import segment_HVI_variables
	test_HVI_marker = segment_HVI_variables("name")

	s = segment_pulse("test", test_HVI_marker)
	from pulse_lib.segments.utility.looping import linspace

	a = tuple()
	b = tuple()
	print(a, b, a+b)
	t2 = linspace(100,500, 20, axis= 0)
	t = linspace(1,50, 10000, name = "test", unit = "test", axis= 0)
	import time
	# s.data_tmp = s.data[0]
	s.add_block(20, 50, 20)
	print(s.data[0].total_time)
	s.add_HVI_marker("test", 15)
	# s.reset_time()
	# s.add_block(20, 30, t)
	# s.wait(10)
	# s.plot_segment()
	# plt.show()
	print(s.setpoints)
	# print(s.loops)
	# print(s.units)
	print(test_HVI_marker.data)