"""
Class that is used to make DC pulses.
"""

import numpy as np
import datetime

from pulse_lib.segments.segment_base import last_edited, segment_base
from pulse_lib.segments.utility.data_handling_functions import loop_controller
from pulse_lib.segments.data_classes.data_pulse import pulse_data
from pulse_lib.segments.data_classes.data_IQ import IQ_data_single
from pulse_lib.segments.utility.setpoint_mgr import setpoint_mgr
from pulse_lib.segments.data_classes.data_pulse_core import base_pulse_element
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
	def add_block(self,start,stop, amplitude):
		'''
		add a block pulse on top of the existing pulse.
		'''

		pulse = base_pulse_element(start + self.data_tmp.start_time,stop + self.data_tmp.start_time, amplitude, amplitude)
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
		
		if keep_amplitude == True:
			pulse = base_pulse_element(start + self.data_tmp.start_time,-1, 0, amplitude)
		else:
			pulse = base_pulse_element(start + self.data_tmp.start_time,stop + self.data_tmp.start_time, 0, amplitude)

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
		if keep_amplitude == True:
			pulse = base_pulse_element(start + self.data_tmp.start_time,-1, start_amplitude, stop_amplitude)
		else:
			pulse = base_pulse_element(start + self.data_tmp.start_time,stop + self.data_tmp.start_time, start_amplitude, stop_amplitude)

		self.data_tmp.add_pulse_data(pulse)

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

	@last_edited
	@loop_controller
	def add_np(self,start, array):
		raise NotImplemented



if __name__ == '__main__':
	import matplotlib.pyplot as plt

	s = segment_pulse("test")
	from pulse_lib.segments.utility.looping import linspace

	a = tuple()
	b = tuple()
	print(a, b, a+b)
	t2 = linspace(100,500, 20, axis= 0)
	t = linspace(1,50, 10000, name = "test", unit = "test", axis= 0)
	import time
	# s.data_tmp = s.data[0]
	s.add_block(0, 50, 100)
	print(s.data[0].total_time)
	# s.reset_time()
	print("test")
	# s.add_block(20, 30, t)
	# s.wait(10)
	s.plot_segment()
	plt.show()
	print(s.setpoints)
	# print(s.loops)
	# print(s.units)