'''
Class that can be used to construct IQ pulses for qubit control applications.

Possible pulses include:
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

'''
import segments_base as seg_base
import numpy as np
from data_handling_functions import loop_controller, IQ_data, linspace




class segment_single_IQ(seg_base.segment_single):
	"""Standard single segment """
	def __init__(self, LO):
		'''
		Args: frequency of the LO (in Hz)
		'''

		super(seg_base.segment_single, self).__init__()
		self.data = np.empty([1], dtype=object)
		self.data[0] = IQ_data(LO)
	
		self.ndim = 0
		self.shape = []
		self.units = []
		self.names = []

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
			't0' : t0,
			't1' : t1,
			'frequency' : freq,
			'amplitude' : amp,
			'phase' : phase,
			}
		self.data_tmp.add_simple_data(data)

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
			't0' : t0,
			't1' : t1,
			'frequency1' : f0,
			'frequency2' : f1,
			'amplitude' : amp,
			}
		self.data_tmp.add_simple_data(data)

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
			't0' : t0,
			't1' : t1,
			'frequency' : freq,
			'amplitude' : amp,
			'phase' : phase,
			'mod' : mod
			}
		self.data_tmp.add_simple_data(data)

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
			't0' : t0,
			't1' : t1,
			'frequency' : freq,
			'amplitude' : amp,
			'phase' : phase,
			'mod' : mod
			}
		self.data_tmp.add_simple_data(data)

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
			't0' : t0,
			't1' : t1,
			'frequency' : freq,
			'amplitude' : amp,
			'phase' : phase,
			'mod' : mod
			}

		self.data_tmp.add_simple_data(data)

	def add_numpy_IQ(self,I, Q):
		raise NotImplemented

	def get_I(self):
		return

	def get_Q(self):
		return
