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
import copy

from pulse_lib.segments.segment_base import last_edited, segment_base
from pulse_lib.segments.utility.data_handling_functions import loop_controller, update_dimension
from pulse_lib.segments.data_classes.data_pulse import pulse_data
from pulse_lib.segments.data_classes.data_IQ import envelope_generator, IQ_data_single, make_chirp
from pulse_lib.segments.data_classes.data_markers import marker_data
from pulse_lib.segments.data_classes.data_generic import data_container



class segment_IQ(segment_base):
	"""
	Standard single segment for IQ purposes
	todo --> add global phase and time shift in the data class instead of this one (cleaner and more generic).
	"""
	def __init__(self, name):
		'''
		Args: 
			name : name of the IQ segment
		Tip, make on of these segments for each qubit. Then you get a very clean implementation of reference frame changes!
		'''
		super().__init__(name, pulse_data() ,segment_type = 'IQ_virtual')
		
		# generator to be set for automated phase compenstation in between pulses. @TODO!!
		self.qubit_freq = 0

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
	def add_MW_pulse(self, t0, t1, amp, freq, phase = 0, AM = None, PM = None):
		'''
		Make a sine pulse (generic constructor)

		Args:
			t0(float) : start time in ns
			t1(float) : stop tiume in ns
			amp (float) : amplitude of the pulse.
			freq(float) : frequency
			phase (float) : phase of the microwave.
			AM ('str/tuple/function') : function describing an amplitude modulation (see examples in pulse_lib.segments.data_classes.data_IQ)
			PM ('str/tuple/function') : function describing an phase modulation (see examples in pulse_lib.segments.data_classes.data_IQ)
		'''
		MW_data = IQ_data_single(t0 + self.data_tmp.start_time, t1 + self.data_tmp.start_time, amp, freq, phase, envelope_generator(AM, PM))

		self.data_tmp.add_MW_data(MW_data)

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
		PM = make_chirp(f0, f1)
		MW_data = IQ_data_single(t0 + self.data_tmp.start_time, t1 + self.data_tmp.start_time, amp, f0, 0, envelope_generator(None, PM))
		
		self.data_tmp.add_MW_data(MW_data)

	def get_IQ_data(self, LO, I_or_Q, image):
		'''
		get I and Q data from the main element.
		Args:
			LO (float): frequency of microwave source
			I_or_Q (str) : 'I'/'Q', denoting if you want the I or the Q channel
			Image : '+'/'-', - if you want the differential signal.
		Returns:
			data (np.ndarray<pulse_data>) : array with the pulsedata objects inside
		'''

		phase_shift = 0
		if I_or_Q == 'Q':
			phase_shift += np.pi/2
		if image == '-':
			phase_shift += np.pi

		local_data = copy.copy(self.data).flatten()
		# downconvert the sigal saved in the data object, so later on, in the real MW source, it can be upconverted again.
		for i in range(len(local_data)):
			local_data[i] = copy.copy(self.data.flat[i])
			local_data[i].shift_MW_phases(phase_shift)
			local_data[i].shift_MW_frequency(LO)
		
		local_data = local_data.reshape(self.data.shape)

		return local_data

	def get_marker_data(self, pre_delay, post_delay):
		'''
		generate markers for the PM of the IQ modulation
		'''
		my_marker_data = update_dimension(data_container(marker_data()), self.shape)
		my_marker_data = my_marker_data.flatten()
		
		# make a flat reference.
		local_data = self.data.flatten()
		
		for i in range(len(local_data)):
			for MW_pulse_info in local_data[i].MW_pulse_data:
				my_marker_data[i].add_marker(MW_pulse_info.start - pre_delay, MW_pulse_info.stop + post_delay)

		my_marker_data.reshape(self.shape)

		return my_marker_data



if __name__ == '__main__':
	import matplotlib.pyplot as plt

	s1 = segment_IQ("test")
	s1.add_MW_pulse(0,1000,1,1e7,0, "flattop")
	s1.reset_time()
	s1.add_chirp(1500,2500,0e7,1e7,1)
	s1.plot_segment(sample_rate = 1e9)
	plt.show()