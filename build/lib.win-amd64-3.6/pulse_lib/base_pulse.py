import numpy as np
import matplotlib.pyplot as plt 
from pulse_lib.segments.segments import segment_container
from pulse_lib.keysight_fx import *
from pulse_lib.sequencer import sequencer
from pulse_lib.keysight.uploader import keysight_uploader
from pulse_lib.keysight.uploader_core.uploader import keysight_upload_module

import uuid
# import qcodes.instrument_drivers.Keysight.SD_common.SD_AWG as keysight_awg
# import qcodes.instrument_drivers.Keysight.SD_common.SD_DIG as keysight_dig

'''
ideas:

# Make a virtual sequence.
# 

'''

class pulselib:
	'''
	Global class that is an organisational element in making the pulses.
	The idea is that you first make individula segments,
	you can than later combine them into a sequence, this sequence will be uploaded
	'''
	def __init__(self):
		# awg channels and locations need to be input parameters.
		self.awg_channels = []
		self.awg_channels_to_physical_locations = dict()
		self.awg_virtual_channels = None
		self.awg_IQ_channels = None
		self.awg_devices = dict()
		self.cpp_uploader = keysight_upload_module()

		self.channel_delays = dict()
		self.channel_delays_computed = dict()
		self.channel_compenstation_limits = dict()

		self.delays = []
		self.convertion_matrix= []
		self.voltage_limits_correction = dict()

		# Keysight properties.
		self.backend = 'keysight'

		self.segments_bin = None
		self.sequencer = None

	def define_channels(self, my_input):
		'''
		define the channels and their location 
		Args:
			my_input (dict): dict of the channel name (str) as key and name of the instrument (as given in add_awgs()) (str) and channel (int) as tuple (e.g. {'chan1' : ('AWG1', 1), ... })
		'''
		self.awg_channels_to_physical_locations = my_input
		self.awg_channels = my_input.keys()
		for i in self.awg_channels:
			self.channel_delays[i] = 0
			self.channel_delays_computed[i] = (0,0)
			self.channel_compenstation_limits[i] = (1500,1500)

	def add_channel_delay(self, delays):
		'''
		Adds to a channel a delay. 
		The delay is added by adding points in front of the first sequence/or 
		just after the last sequence. The first value of the sequence will be 
		taken as an extentsion point.

		Args:
			delays: dict, e.g. {'P1':20, 'P2':16} delay P1 with 20 ns and P2 with 16 ns

		Returns:
			0/Error
		'''
		for i in delays.items():
			if i[0] in self.awg_channels:
				self.channel_delays[i[0]] = i[1]
			else:
				raise ValueError("Channel delay error: Channel '{}' does not exist. Please provide valid input".format(i[0]))

		self.__process_channel_delays()
		return 0

	def add_channel_compenstation_limits(self, limits):
		'''
		add voltage limitations per channnel that can be used to make sure that the intragral of the total voltages is 0.
		Args:
			limits (dict) : dict with limits e.g. {'B0':(-100,500), ... }
		Returns:
			None
		'''
		for i in limits.items():
			if i[0] in self.awg_channels:
				self.channel_compenstation_limits[i[0]] = i[1]
			else:
				raise ValueError("Channel voltage compenstation error: Channel '{}' does not exist. Please provide valid input".format(i[0]))


	def add_awgs(self, name, awg):
		'''
		add a awg to the library
		Args:
			name (str) : name you want to give to a peculiar AWG
			awg (object) : qcodes object of the concerning AWG
		'''
		self.awg_devices[name] =awg
		# self.cpp_uploader.add_awg_module(name, awg)

	def add_virtual_gates(self, virtual_gates):
		'''
		define virtual gates for the gate set.
		Args:
			virtual_gates (dict): needs to have the following keys:
				'virtual_gates_names_virt' : should constain a list with the channel names of the virtual gates
				'virtual_gates_names_real' : should constain a list with the channel names of the read gates (should be as long as the virtual ones)
				'virtual_gate_matrix' : numpy array representing the virtual gate matrix
		'''
		self.awg_virtual_channels = virtual_gates

	def update_virtual_gate_matrix(self, new_matrix):
		raise NotImplemented

	def add_IQ_virt_channels(self, IQ_virt_channels):
		'''
		function to define virtual IQ channels (a channel that combined the I and Q channel for MW applications):
		Args:
			IQ_virt_channels (dict): a dictionary that needs to contain the following keys:
				'vIQ_channels' : list of names of virtual IQ channels
				'r_IQ_channels': list of list, where in each list the two reference channels (I and Q) are denoted (see docs for example).
				'LO_freq'(function/double) : local oscillating frequency of the source. Will be used to do automaticcally convert the freq a SSB signal.  
		'''
		self.awg_IQ_channels = IQ_virt_channels

	def finish_init(self):
		# function that finishes the initialisation
		# TODO rewrite, so this function is embedded in the other ones.
		self.uploader = keysight_uploader(self.awg_devices, self.cpp_uploader, self.awg_channels, self.awg_channels_to_physical_locations , self.channel_delays_computed, self.channel_compenstation_limits)

	def mk_segment(self):
		'''
		generate a new segment.
		Returns:
			segment (segment_container) : returns a container that contains all the previously defined gates.
		'''
		return segment_container(self.awg_channels, self.awg_virtual_channels, self.awg_IQ_channels)

	def mk_sequence(self,seq):
		'''
		seq: list of list, 
			e.g. [ ['name segment 1' (str), number of times to play (int), prescale (int)] ]
			prescale (default 0, see keysight manual) (not all awg's will support this).
		'''
		seq_obj = sequencer(self.uploader, self.voltage_limits_correction)
		seq_obj.add_sequence(seq)
		return seq_obj

	def __process_channel_delays(self):
		'''
		Makes a variable that contains the amount of points that need to be put before and after when a upload is performed.
		'''
		self.channel_delays_computed = dict()

		for channel in self.channel_delays:
			self.channel_delays_computed[channel] = (self.__get_pre_delay(channel), self.__get_post_delay(channel))


	def __calculate_total_channel_delay(self):
		'''
		function for calculating how many ns time there is a delay in between the channels.
		Also support for negative delays...

		returns:
			tot_delay (the total delay)
			max_delay (hight amount of the delay)
		'''

		delays =  np.array( list(self.channel_delays.values()))
		tot_delay = np.max(delays) - np.min(delays)

		return tot_delay, np.max(delays)

	def __get_pre_delay(self, channel):
		'''
		get the of ns that a channel needs to be pushed forward/backward.
		returns
			pre-delay : number of points that need to be pushed in from of the segment
		'''
		tot_delay, max_delay = self.__calculate_total_channel_delay()
		max_pre_delay = tot_delay - max_delay
		delay = self.channel_delays[channel]
		return -(delay + max_pre_delay)

	def __get_post_delay(self, channel):
		'''
		get the of ns that a channel needs to be pushed forward/backward.
		returns
			post-delay: number of points that need to be pushed after the segment
		'''
		tot_delay, max_delay = self.__calculate_total_channel_delay()
		delay = self.channel_delays[channel]

		return -delay + max_delay




if __name__ == '__main__':
	p = pulselib()

	
	class AWG(object):
		"""docstring for AWG"""
		def __init__(self, name):
			self.name = name
			self.chassis = 0
			self.slot = 0
			self.type = "DEMO"

	AWG1 = AWG("AWG1")
	AWG2 = AWG("AWG2")
	AWG3 = AWG("AWG3")
	AWG4 = AWG("AWG4")
		
	# add to pulse_lib
	p.add_awgs('AWG1',AWG1)
	p.add_awgs('AWG2',AWG2)
	p.add_awgs('AWG3',AWG3)
	p.add_awgs('AWG4',AWG4)

	# define channels
	awg_channels_to_physical_locations = dict({'B0':('AWG1', 1), 'P1':('AWG1', 2),
		'B1':('AWG1', 3), 'P2':('AWG1', 4),
		'B2':('AWG2', 1), 'P3':('AWG2', 2),
		'B3':('AWG2', 3), 'P4':('AWG2', 4),
		'B4':('AWG3', 1), 'P5':('AWG3', 2),
		'B5':('AWG3', 3), 'G1':('AWG3', 4),
		'I_MW':('AWG4', 1), 'Q_MW':('AWG4', 2),	
		'M1':('AWG4', 3), 'M2':('AWG4', 4)})
		
	p.define_channels(awg_channels_to_physical_locations)

	# format : dict of channel name with delay in ns (can be posive/negative)
	p.add_channel_delay({'I_MW':50, 'Q_MW':50, 'M1':20, 'M2':-25, })

	awg_virtual_gates = {'virtual_gates_names_virt' :
		['vP1','vP2','vP3','vP4','vP5','vB0','vB1','vB2','vB3','vB4','vB5'],
			'virtual_gates_names_real' :
		['P1','P2','P3','P4','P5','B0','B1','B2','B3','B4','B5'],
			'virtual_gate_matrix' : np.eye(11)
	}
	p.add_virtual_gates(awg_virtual_gates)

	awg_IQ_channels = {'vIQ_channels' : ['qubit_1','qubit_2'],
				'rIQ_channels' : [['I_MW','Q_MW'],['I_MW','Q_MW']],
				'LO_freq' :[2e9, 1e9]
				# do not put the brackets for the MW source
				# e.g. MW_source.frequency
				}
		
	p.add_IQ_virt_channels(awg_IQ_channels)

	p.finish_init()

	seg  = p.mk_segment()
	seg2 = p.mk_segment()
	seg3 = p.mk_segment()

	seg.vP1.add_block(0,10,1)


	# B0 is the barrier 0 channel
	# adds a linear ramp from 10 to 20 ns with amplitude of 5 to 10.
	seg.B0.add_pulse([[10.,0.],[10.,5.],[20.,10.],[20.,0.]])
	# add a block pulse of 2V from 40 to 70 ns, to whaterver waveform is already there
	seg.B0.add_block(40,70,2)
	# just waits (e.g. you want to ake a segment 50 ns longer)
	seg.B0.wait(50)
	# resets time back to zero in segment. Al the commannds we run before will be put at a negative time.
	seg.B0.reset_time()
	# this pulse will be placed directly after the wait()
	seg.B0.add_block(0,10,2)

