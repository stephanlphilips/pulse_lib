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
class linspace():
	"""docstring for linspace"""
	def __init__(self, start, stop, n_steps = 50, name = "undefined", unit = 'a.u.', axis = -1):
		self.data = np.linspace(start, stop, n_steps)
		self.name = name
		self.unit = unit
		self.axis = axis
	def __len__(self):
		return len(self.data)

def loop_controller(func):
	'''
	Checks if there are there are parameters given that are loopable.

	If loop:
		* then check how many new loop parameters on which axis
		* extend data format to the right shape (simple python list used).
		* loop over the data and add called function

	if no loop, just apply func on all data (easy)
	'''
	def wrapper(*args, **kwargs):
		obj = args[0]

		loop_info_args = []
		loop_info_kwargs = []

		for i in range(1,len(args)):
			if type(args[i]) == linspace : 
				info = {
				'nth_arg': i,
				'name': args[i].name,
				'len': len(args[i]),
				'axis': args[i].axis,
				'data' : args[i].data}
				loop_info_args.append(info)

		for i in range(1,len(kwargs)):
			if type(kwargs.values()[i]) == linspace : 
				info = {
				'nth_arg': i,
				'name': kwargs.values()[i].name,
				'len': len(kwargs.values()[i]),
				'axis': kwargs.values()[i].axis,
				'data' : kwargs.values()[i].data}
				loop_info_kwargs.append(info)
		
		
		print(loop_info_args, loop_info_kwargs)

	return wrapper

class segment_single_IQ(seg_base.segment_single):
	"""Standard single segment """
	def __init__(self, LO):
		'''
		Args: frequency of the LO (in Hz)
		'''

		super(seg_base.segment_single, self).__init__()
		self.data = IQ_data(LO)
		self.LO = LO
	
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
		self.IQ_data.add_simple_data(data)

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
		self.IQ_data.add_simple_data(data)

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
		self.IQ_data.add_simple_data(data)

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
		self.IQ_data.add_simple_data(data)

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

		self.IQ_data.add_simple_data(data)

	def add_numpy_IQ(self,I, Q):
		raise NotImplemented

	def get_I(self):
		return

	def get_Q(self):
		return

class IQ_data():
	"""class that manages the data used for generating IQ data"""
	def __init__(self, LO):
		self.LO = LO
		self.simple_IQ_data = []
		self.MOD_IQ_data = []
		self.numpy_IQ_data = []

	def add_simple_data(self, input_dict):
		self.simple_IQ_data.append(input_dict)
	
	def add_mod_data (self, input_dict):
		self.simple_IQ_data.append(input_dict)

	def add_numpy_IQ(self, input_dict):
		self.numpy_IQ_data.append(input_dict)




t = segment_single_IQ(1e9)
t.add_sin(0,10e-9, linspace(0,100, name="Freq", axis=1), linspace(0,100, name="Amp", axis=0), 0)
