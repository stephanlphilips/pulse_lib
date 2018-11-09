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
from data_handling_functions import loop_controller, linspace
from data_classes import IQ_data




class segment_single_IQ(seg_base.segment_single):
	"""
	Standard single segment for IQ purposes
	todo --> add global phase and time shift in the data class instead of this one (cleaner and more generic).
	"""
	def __init__(self, LO):
		'''
		Args: frequency of the LO (in Hz). 
		Tip, make on of these segments for each qubit. Then you get a very clean implementation of reference frame changes!
		'''

		super(seg_base.segment_single, self).__init__()
		self.data = np.empty([1], dtype=object)
		self.data[0] = IQ_data(LO)
		
		self.ndim = 0
		self.shape = []
		self.units = []
		self.names = []

	@loop_controller
	def add_global_phase(self,phase):
		"""
		global shift in phase for all the pulses that will follow.
		Args:
			phase (double) : phase in radians

		Use this function to apply a reference change for the qubit.
		"""
		print("adding_phase")
		self.data_tmp.global_phase += phase

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

	@loop_controller
	def add_numpy_IQ(self,I, Q):
		raise NotImplemented

	def get_I(self):
		'''
		get I and Q data from the main element.
		'''
		local_data = np.flatten(local_data)
		I_data = np.empty(local_data.shape, dtype=object)
		
		for i in range(len(I_data)):
			I_data[i] =  local_data[i].get_I()
		
		I_data.reshape(self.data.shape)

		return I_data
	
	def get_Q(self):
		local_data = np.flatten(local_data)
		Q_data = np.empty(local_data.shape, dtype=object)

		for i in range(len(Q_data)):
			Q_data[i] =  local_data[i].get_I()
		
		Q_data.reshape(self.data.shape)

		return Q_data



seg = segment_single_IQ(1e9)
seg.add_sin(0, linspace(5,105,100, axis = 0), 100, 1.1e9)

seg.reset_time()
seg.add_global_phase(0.1)
seg.add_sin(0, linspace(5,105,100, axis = 0), 1.1e9, 10)

print(seg.data[0].simple_IQ_data)
print(seg.data[20].simple_IQ_data)
print(seg.data[80].simple_IQ_data)