import sys
sys.path.append('C:\Program Files (x86)\Keysight\SD1\Libraries\Python')
sys.path.append("C:\\V2_code\\Qcodes")

import numpy as np
import matplotlib.pyplot as plt 
from segments import *
from keysight_fx import *
import uuid
import qcodes.instrument_drivers.Keysight.SD_common.SD_AWG as keysight_awg
import qcodes.instrument_drivers.Keysight.SD_common.SD_DIG as keysight_dig

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
		self.awg_channels = ['P1','P2','P3','P4','P5','B0','B1','B2','B3','B4','B5','G1','I_MW', 'Q_MW', 'M1', 'M2']
		self.awg_channels_to_physical_locations = dict({'B0':('AWG1', 1),
															'P1':('AWG1', 2),
															'B1':('AWG1', 3),
															'P2':('AWG1', 4),
															'B2':('AWG2', 1),
															'P3':('AWG2', 2),
															'B3':('AWG2', 3),
															'P4':('AWG2', 4),
															'B4':('AWG3', 1),
															'P5':('AWG3', 2),
															'B5':('AWG3', 3),
															'G1':('AWG3', 4),
															'I_MW':('AWG4', 1),
															'Q_MW':('AWG4', 2),
															'M1':('AWG4', 3),
															'M2':('AWG4', 4)})
		self.awg_channels_kind = []

		self.awg_virtual_channels = {'virtual_gates_names_virt' : ['vP1','vP2','vP3','vP4','vP5','vB0','vB1','vB2','vB3','vB4','vB5'],
									 'virtual_gates_names_real' : ['P1','P2','P3','P4','P5','B0','B1','B2','B3','B4','B5'],
									 'virtual_gate_matrix' : np.eye(11)}
		self.awg_virtual_channels['virtual_gate_matrix'][0,1] = 0.1
		self.awg_virtual_channels['virtual_gate_matrix'][0,2] = 0.1

		# Not implemented
		self.awg_markers =['mkr1', 'mkr2', 'mkr3' ]
		self.awg_markers_to_location = []

		self.channel_delays = dict()
		for i in self.awg_channels:
			self.channel_delays[i] = 0
		
		self.delays = []
		self.convertion_matrix= []
		self.segments_bin = segment_bin(self.awg_channels, self.awg_virtual_channels)

		awg1 = keysight_awg.SD_AWG('awg1', chassis = 0, slot= 2, channels = 4, triggers= 8)
		awg2 = keysight_awg.SD_AWG('awg2', chassis = 0, slot= 3, channels = 4, triggers= 8)
		awg3 = keysight_awg.SD_AWG('awg3', chassis = 0, slot= 4, channels = 4, triggers= 8)
		awg4 = keysight_awg.SD_AWG('awg4', chassis = 0, slot= 5, channels = 4, triggers= 8)
		
		# Keysight properties.
		self.backend = 'keysight'
		self.awg = keysight_AWG(self.segments_bin, self.awg_channels_to_physical_locations, self.awg_channels, self.channel_delays)
		self.awg.add_awg('AWG1',awg1)
		self.awg.add_awg('AWG2',awg2)
		self.awg.add_awg('AWG3',awg3)
		self.awg.add_awg('AWG4',awg4)


		self.sequencer =  sequencer(self.awg, self.channel_delays, self.segments_bin)

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
		self.awg.add_awg(name, awg)

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

	def __init__(self, channels, virtual_gate_matrix=None):
		self.segment = []
		self.channels = channels
		self.virtual_gate_matrix = virtual_gate_matrix
		return	

	def new(self,name):
		if self.exists(name):
			raise ValueError("sement with the name : % \n alreadt exists"%name)
		self.segment.append(segment_container(name,self.channels, self.virtual_gate_matrix))
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
p.add_channel_delay({'B4':-3,})
p.add_channel_delay({'M2':-115,})

#%%
p.sequencer =  sequencer(p.awg, p.channel_delays, p.segments_bin)

seg = p.mk_segment('INIT')
	
seg2 = p.mk_segment('Manip')
seg3 = p.mk_segment('Readout')




# seg.B4.add_block(0,10,1)
# seg.B4.wait(100)
# seg.B4.add_block(0,10,1)

# seg.B4.add_block(200,300,1)
seg.B0.add_block(0, 10, 1.5)
# seg.B0.add_block(10000, 10010, 1)
# seg.B0.add_block(10010, 10112, 0)
seg.B0.wait(10000)
seg.B4.add_block(0, 10, -1.5)
seg.B4.wait(10000)
# amp = np.linspace(0,1,50)
# period = 1000
# t = 0
# for i in range(50):
# 	seg.B4.add_block(t, t+period, amp[i])
# 	t+= period
# seg.B4.repeat(5)

# amp = np.linspace(0,1,50)
# period = 1000*50
# t = 0
# for i in range(50):
# 	seg.I_MW.add_block(t, t+period, amp[i])
# 	t+= period

# seg.G1.add_block(100, 130, 1)
# seg.M2.add_block(100, 130, 1)

# seg.B0.add_block(20,205,1)
# seg.B0.add_block(205,285,-1)sd
# seg.B0.add_block(20,205,1)
# seg.B0.add_block(20,50000,1)


# segs = [seg.B0, seg.P1, seg.B1, seg.P2]

# amp = 0.25
# for i in segs:
# 	i.add_pulse([[0,-amp]])
# 	# i.add_block(20,25, -amp + 3)
# 	# i.add_block(60,65, -amp + 2)
# 	# i.add_block(100,105, -amp + 1)
# 	# i.add_block(140,145, -amp + 0.5)
# 	# i.add_block(180,185, -amp + 0.2)
# 	t = 10
# 	for j in range(1,50):
# 		i.add_block(t,t+j*2,amp)
# 		t = t+j*2
# 		t = t + 50

# 	i.add_pulse([[10000,-amp]])


# amp = 1

# seg.P1.add_pulse([
#         [0,-amp],
#         [10,amp],
#         [9000,amp],
#         [9000,-amp],
#         [14000,-amp]])

# seg.B4.add_pulse([
#         [0,-amp],
#         [10,amp],
#         [9000,amp],
#         [9000,-amp],
#         [14000,-amp]])


# t = 0 
# amp = 0.1
# seg.B4.add_pulse([
#         [0,-amp]])

# for i in range(1,50):
# 	seg.B4.add_block(t,t+i*2,amp)
# 	t = t+i*2
# 	t = t + 50


# t = 0
# seg.P1.add_pulse([
#         [0,-amp]])

# for i in range(1,50):
# 	seg.P1.add_block(t,t+i*2,amp)
# 	t = t+i*2
# 	t = t + 50
# seg.B4.add_pulse([
#         [0,-1.5],
#         [40,-1.5], 
#         [40,1.5],
#         [80,0],
#         [80,-0.75],
#         [90,-0.75]])
# seg.B4.add_block(100,110,-0.75+0.15)
# seg.B4.add_block(120,130,-0.75+0.1)
# seg.B4.add_block(140,150,-0.75+0.05)
# seg.B4.add_pulse([
#         [200,0],
#         [210,1],
#         [210,-0.2],
#         [240,0],
#         [275,1.3],
#         [275,-0.75],
#         [300,-0.75],
#         [300,-0.5],
#         [310,-0.5],
#         [310,0]])


# seg.B4.add_block(10,20000,0.3)
# seg.P5.add_block(10,20000,-0.3)
# seg.B5.add_block(10,20000,-0.3)
# seg.G1.add_block(10,20000,0.3)
# append functions?
# seg.P1.add_block(2,5,-1)
# seg.P1.add_pulse([[100,0.5]
# 				 ,[800,0.5],
# 				  [1400,0]])

# seg.B2.add_block(2,5,-1)
# seg.B2.add_pulse([[20,0],[30,0.5], [30,0]])
# seg.B2.add_block(40,70,1)
# seg.B2.add_pulse([[70,0],
# 				 [80,0],
# 				 [150,0.5],
# 				 [150,0]])

# seg.B4.add_block(2,5,1)
# seg.B4.add_block(2,10,1)
# seg.M2.wait(50)
# seg.M2.plot_sequence()
# seg.B0.repeat(20)
# seg.B0.wait(20)
# print(seg.B0.my_pulse_data)
# seg.reset_timevoltage_range_reset_needed()
# seg.B2.add_block(30,60,1)
# seg.B2.add_block(400,800,0.5)
# seg.B2.add_block(1400,1500,0.5)
# seg.B1.plot_sequence()
# seg.M2.add_pulse([[20,0.2],[30,0]])
# seg.M2.add_block(30,60,1)
# seg.M2.wait(2000)
# seg.M2.add_block(90,120,1)
# seg.M2.plot_sequence()
# seg.M2.add_block(400,800,0.5)
# seg.M2.add_block(1400,1500,0.5)

# seg2.B2.add_block(30,60,0)
# seg2.B2.add_block(400,2000,0.1)
# seg2.P1.add_block(30,60,0)
# seg2.P1.add_block(400,800,0.5)
# seg2.B0.add_block(30,60,0.1)
# seg2.B0.add_block(400,800,0.1)
# seg2.B0.wait(2000)
# seg3.B5.add_block(30,600,0.1)
# seg3.B5.wait(2000)
p.show_sequences()

SEQ = [['INIT', 1, 0]]

p.add_sequence('mysequence', SEQ)

p.start_sequence('mysequence')

SEQ2 = [['INIT', 1, 0], ['Manip', 1, 0], ['Readout', 1, 0] ]

# p.add_sequence('mysequence2', SEQ2)

# p.start_sequence('mysequence2')
# insert in the begining of a segment
# seg.insert_mode()
# seg.clear()

# # class channel_data_obj():
# #     #object containing the data for a specific channels
# #     #the idea is that all the data is parmeterised and will be constuceted whenever the function is called.

# #     self.my_data_array = np.empty()
    
# #     add_data

# # class block_pulses:
# #     # class to make block pulses


# # how to do pulses
# # -> sin?
# # -> pulses?
# # -> step_pulses

# # p = pulselin()

# # seg = pulselib.mk_segment('manip')
# # seg.p1.add_pulse(10,50, 20, prescaler= '1')
# # seg.p3.add_pulse(12,34, 40,)
# # seg.k2.add_pulse_advanced([pulse sequence])
# # seg.add_np(array, tstart_t_stop
# # seg.p5.add_sin(14,89, freq, phase, amp)

# # pulse

# import datetime
# print(datetime.datetime.utcfromtimestamp(0))