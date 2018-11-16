import sys
sys.path.append('C:\Program Files (x86)\Keysight\SD1\Libraries\Python')
sys.path.append("C:\\V2_code\\Qcodes")

import numpy as np
import matplotlib.pyplot as plt 
from pulse_lib.segments.segments import segment_container
from pulse_lib.keysight_fx import *
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
		self.awg_channels_to_physical_locations = []
		self.awg_virtual_channels = None
		self.awg_IQ_channels = None
		self.awg_devices = []
		
		self.channel_delays = dict()
		
		
		self.delays = []
		self.convertion_matrix= []

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

		return 0

	def add_awgs(self, name, awg):
		'''
		add a awg to the library
		Args:
			name (str) : name you want to give to a peculiar AWG
			awg (object) : qcodes object of the concerning AWG
		'''
		self.awg_channels.append([name, awg])

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
		self.segments_bin = segment_bin(self.awg_channels, self.awg_virtual_channels, self.awg_IQ_channels)
		self.awg = keysight_AWG(self.segments_bin, self.awg_channels_to_physical_locations, self.awg_channels, self.channel_delays)
		for i in self.awg_devices:
			self.awg.add_awg(i[0], i[1])
		self.sequencer =  sequencer(self.awg, self.channel_delays, self.segments_bin)

	def mk_segment(self, name):
		return self.segments_bin.new(name)

	def get_segment(self, name):
		return self.segments_bin.get(name)

	def add_sequence(self,name,seq):
		'''
		name: name for the sequence, if name already present, it will be overwritten.
		seq: list of list, 
			e.g. [ ['name segment 1' (str), number of times to play (int), prescale (int)] ]
			prescale (default 0, see keysight manual) (not all awg's will support this).
		'''
		self.sequencer.add_sequence(name, seq)

	def start_sequence(self, name):
		self.sequencer.start_sequence(name)
		
	def show_sequences(self):
		self.segments_bin.print_segments_info()


class segment_bin():

	def __init__(self, channels, virtual_gate_matrix=None, IQ_virt_chan=None):
		self.segment = []
		self.channels = channels
		self.virtual_gate_matrix = virtual_gate_matrix
		self.IQ_virt_chan = IQ_virt_chan	

	def new(self,name):
		if self.exists(name):
			raise ValueError("sement with the name : % \n alreadt exists"%name)
		self.segment.append(segment_container(name,self.channels, self.virtual_gate_matrix, self.IQ_virt_chan))
		return self.get_segment(name)

	def get_segment(self, name):
		for i in self.segment:
			if i.name == name:
				return i
		raise ValueError("segment not found :(")

	def used(self, name):
		''' check if a segment is used  (e.g. that a leas one element is used) 
		name is a string.
		'''
		seg = self.get_segment(name)
		
		if seg.total_time == 0:
			return False
		return True	

	def print_segments_info(self):
		mystring = "Sequences\n"
		for i in self.segment:
			mystring += "\tname: " + i.name + "\n"
		print(mystring)

	def exists(self, name):
		for i in self.segment:
			if i.name ==name:
				return True
		return False

	def get_time_segment(self, name):
		return self.get_segment(name).total_time



class sequencer():
	def __init__(self, awg_system, channel_delays, segment_bin):
		self.awg = awg_system
		self.segment_bin = segment_bin
		self.channels = segment_bin.channels
		self.channel_delays = channel_delays
		self.sequences = dict()

	def add_sequence(self, name, sequence):
		self.sequences[name] = sequence

	def start_sequence(self, name):
		self.get_sequence_upload_data(name)
		self.awg.upload(self.sequences[name], self.get_sequence_upload_data(name))
		self.awg.start()

	def get_sequence_upload_data(self, name):
		'''
		Function that generates sequence data per channel.
		It will also assign unique id's to unique sequences (sequence that depends on the time of playback). -> mainly important for iq mod purposes.
		structure:
			dict with key of channel:
			for each channels list of sequence:
				name of the segments,
				number of times to play
				uniqueness -> segment is reusable?
				identifiers for marking differnt locations in the ram of the awg.
		
		'''
		upload_data = dict()
		# put in a getter to make sure there is no error -- it exists...
		seq = self.sequences[name]

		for chan in self.channels:
			sequence_data_single_channel = []
			num_elements = len(seq)

			for k in range(len(seq)):
				segment_play_info = seq[k]

				# if starting segment or finishing segment, here there should be added the delay info.
				pre_delay, post_delay = (0,0)

				if k == 0:
					pre_delay = self.get_pre_delay(chan)
				if k == len(seq)-1:
					post_delay = self.get_post_delay(chan)

				if pre_delay!=0 or post_delay!=0:
					rep = segment_play_info[1]
					segment_play_info[1] = 1
					input_data = self.generate_input_data(segment_play_info, chan, pre_delay, post_delay)
					sequence_data_single_channel.append(input_data)

					# If only one, go to next segment in the sequence.
					if rep == 1 :
						continue
					else:
						segment_play_info[1] = rep -1

				sequence_data_single_channel.append(self.generate_input_data(segment_play_info, chan))

			upload_data[chan] = sequence_data_single_channel

		return upload_data


	def generate_input_data(self, segment_play_info, channel, pre_delay=0, post_delay=0):
		'''
		function that will generate a dict that defines the input data, this will contain all the neccesary info to upload the segment.
		returns:
			dict with sequence info for a cerain channel (for parameters see the code).
		'''
		input_data = {'segment': segment_play_info[0], 
						'segment_name': self.make_segment_name(segment_play_info[0], pre_delay, post_delay),
						'ntimes': segment_play_info[1],
						'prescaler': segment_play_info[2],
						'pre_delay': pre_delay,
						'post_delay': post_delay}
		unique = getattr(self.segment_bin.get_segment(segment_play_info[0]), channel).unique
		input_data['unique'] = unique
		# Make unique uuid's for each segment
		if unique == True:
			input_data['identifier'] = [uuid.uuid4() for i in range(segment_play_info[1])]

		return input_data

	def make_segment_name(self, segment, pre_delay, post_delay):
		'''
		function that makes the name of the segment that is delayed.
		Note that if the delay is 0 there should be no new segment name.
		'''
		segment_name = segment
		
		if pre_delay!=0 or post_delay!= 0:
			segment_name = segment + '_' + str(pre_delay) + '_' + str(post_delay)

		return segment_name

	def calculate_total_channel_delay(self):
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

	def get_pre_delay(self, channel):
		'''
		get the of ns that a channel needs to be pushed forward/backward.
		returns
			pre-delay : number of points that need to be pushed in from of the segment
		'''
		tot_delay, max_delay = self.calculate_total_channel_delay()
		max_pre_delay = tot_delay - max_delay
		delay = self.channel_delays[channel]
		return delay + max_pre_delay

	def get_post_delay(self, channel):
		'''
		get the of ns that a channel needs to be pushed forward/backward.
		returns
			post-delay: number of points that need to be pushed after the segment
		'''
		tot_delay, max_delay = self.calculate_total_channel_delay()
		delay = self.channel_delays[channel]
		print(tot_delay, max_delay, delay)
		print("post_delay", delay - (tot_delay - max_delay))
		return -delay + max_delay

p = pulselib()

awg1 = None
awg2 = None
awg3 = None
awg4 = None

# add to pulse_lib
p.add_awgs('AWG1',awg1)
p.add_awgs('AWG2',awg2)
p.add_awgs('AWG3',awg3)
p.add_awgs('AWG4',awg4)

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

seg  = p.mk_segment('INIT')
seg2 = p.mk_segment('Manip')
seg3 = p.mk_segment('Readout')

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