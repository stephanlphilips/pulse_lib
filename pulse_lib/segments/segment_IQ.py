'''
Class that can be used to construct IQ pulses for qubit control applications.

Possible pulses include:o
* standard block pulse
* chirped pulse for adiabatic spin operation
* modulated pulsed.

As data format we will use a class to store
* type (std, chrip, mod)
* t0
* te
* freq1
* freq2 (opt)
* amp
* phase 

TODO : change dicts to keep the data to an object!!
'''
import numpy as np
import datetime

from pulse_lib.segments.segment_base import last_edited, segment_base
from pulse_lib.segments.utility.data_handling_functions import loop_controller
from pulse_lib.segments.data_classes.data_IQ import IQ_data



class segment_IQ(segment_base):
	"""
	Standard single segment for IQ purposes
	todo --> add global phase and time shift in the data class instead of this one (cleaner and more generic).
	"""
	def __init__(self, name, LO):
		'''
		Args: frequency of the LO (in Hz). 
		Tip, make on of these segments for each qubit. Then you get a very clean implementation of reference frame changes!
		'''
		super().__init__(name, IQ_data(LO) ,segment_type = 'IQ_virtual')

	@loop_controller
	def add_global_phase(self,phase):
		"""
		global shift in phase for all the pulses that will follow.
		Args:
			phase (double) : phase in radians

		Use this function to apply a reference change for the qubit.
		"""
		self.data_tmp.global_phase += phase

	@last_edited
	@loop_controller
	def add_sin(self, t0, t1, freq, amp, phase = 0):
		'''
		Add simple sine to the segment.
		Args:
			t0(float) : start time in ns
			t1(float) : stop tiume in ns
			freq(float) : frequency
			amp (float) : amplitude of the pulse.
			phase (float) : phase of the microwave.
		'''
		data = {
			'type' : 'std',
			'start_time' : t0 + self.data_tmp.start_time,
			'stop_time' : t1 + self.data_tmp.start_time,
			'frequency' : freq,
			'amplitude' : amp,
			'phase' : phase + self.data_tmp.global_phase,
			}
		self.data_tmp.add_simple_data(data)

	@last_edited
	@loop_controller
	def add_chirp(self, t0, t1, f0, f1, amp):
		'''
		Add chirp to the segment.
		Args:
			t0(float) : start time in ns
			t1(float) : stop tiume in ns
			f0(float) : start frequency
			f1 (float) : stop frequency
			amp (float) : amplitude of the pulse.
		'''
		data = {
			'type' : 'chirp',
			'start_time' : t0 + self.data_tmp.start_time,
			'stop_time' : t1 + self.data_tmp.start_time,
			'frequency1' : f0,
			'frequency2' : f1,
			'amplitude' : amp,
			}
		self.data_tmp.add_simple_data(data)

	@last_edited
	@loop_controller
	def add_AM_sin(self, t0, t1, freq, amp, phase, mod=None):
		'''
		Add amplitude modulation.
		Args:
			t0(float) : start time in ns
			t1(float) : stop tiume in ns
			freq(float) : frequency of the pulse (total frequency, not offet frequency from central frequency)
			amp (float) : amplitude of the applied signal in the unit you set the pulse lib
			mod (np.ndarray) : modualtion of the sin(should have the same number of points as the time between t0 and t1 (use same sample rate))
		'''
		data = {
			'type' : 'AM_mod',
			'start_time' : t0 + self.data_tmp.start_time,
			'stop_time' : t1 + self.data_tmp.start_time,
			'frequency' : freq,
			'amplitude' : amp,
			'phase' : phase + self.data_tmp.global_phase,
			'mod' : mod
			}
		self.data_tmp.add_simple_data(data)

	@last_edited
	@loop_controller
	def add_FM_sin(self, t0, t1, freq, amp, phase, mod=None):
		'''
		Add frequency modulation.
		Args:
			t0(float) : start time in ns
			t1(float) : stop tiume in ns
			freq(float) : frequency of the pulse (total frequency, not offet frequency from central frequency)
			amp (float) : amplitude of the applied signal in the unit you set the pulse lib
			mod (np.ndarray) : modualtion of the sin(should have the same number of points as the time between t0 and t1 (use same sample rate))
		'''
		data = {
			'type' : 'FM_mod',
			'start_time' : t0 + self.data_tmp.start_time,
			'stop_time' : t1 + self.data_tmp.start_time,
			'frequency' : freq,
			'amplitude' : amp,
			'phase' : phase + self.data_tmp.global_phase,
			'mod' : mod
			}
		self.data_tmp.add_simple_data(data)

	@last_edited
	@loop_controller
	def add_PM_sin(self, t0, t1, freq, amp, phase, mod=None):
		'''
		Add phase modulation.
		Args:
			t0(float) : start time in ns
			t1(float) : stop tiume in ns
			freq(float) : frequency of the pulse (total frequency, not offet frequency from central frequency)
			amp (float) : amplitude of the applied signal in the unit you set the pulse lib
			mod (np.ndarray) : modualtion of the sin(should have the same number of points as the time between t0 and t1 (use same sample rate))
		'''
		data = {
			'type' : 'PM_mod',
			'start_time' : t0 + self.data_tmp.start_time,
			'stop_time' : t1 + self.data_tmp.start_time,
			'frequency' : freq,
			'amplitude' : amp,
			'phase' : phase + self.data_tmp.global_phase,
			'mod' : mod
			}

		self.data_tmp.add_simple_data(data)

	@last_edited
	@loop_controller
	def add_numpy_IQ(self,I, Q):
		raise NotImplemented

	def get_IQ_data(self, I_or_Q):
		'''
		get I and Q data from the main element.
		Args:
			I_or_Q (str) : 'I'/'Q', denoting if you want the I or the Q channel
		Returns:
			data (np.ndarray) : array with the objects inside
		'''
		local_data = self.data.flatten()
		data = np.empty(local_data.shape, dtype=object)
		
		for i in range(len(data)):
			data[i] =  local_data[i].get_IQ_data(I_or_Q)
		
		data = data.reshape(self.data.shape)
		return data


if __name__ == '__main__':
	import matplotlib.pyplot as plt

	s1 = segment_IQ("test", 1e9)
	s1.add_sin(20,50,1e6,1)


	# s1.plot_segment()
	# s2.plot_segment()
	# s3.plot_segment()
	# plt.legend()
	# plt.show()
		