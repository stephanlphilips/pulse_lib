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
		# Not implemented
		self.awg_markers =['mkr1', 'mkr2', 'mkr3' ]
		self.awg_markers_to_location = []
		self.delays = []
		self.convertion_matrix= []
		self.segments_bin = segment_bin(self.awg_channels)

		awg1 = keysight_awg.SD_AWG('awg1', chassis = 0, slot= 2, channels = 4, triggers= 8)
		awg2 = keysight_awg.SD_AWG('awg2', chassis = 0, slot= 3, channels = 4, triggers= 8)
		awg3 = keysight_awg.SD_AWG('awg3', chassis = 0, slot= 4, channels = 4, triggers= 8)
		awg4 = keysight_awg.SD_AWG('awg4', chassis = 0, slot= 5, channels = 4, triggers= 8)
		
		# Keysight properties.
		self.backend = 'keysight'
		self.awg = keysight_AWG(self.segments_bin, self.awg_channels_to_physical_locations, self.awg_channels)
		self.awg.add_awg('AWG1',awg1)
		self.awg.add_awg('AWG2',awg1)
		self.awg.add_awg('AWG3',awg1)
		self.awg.add_awg('AWG4',awg1)


		self.sequencer =  sequencer(self.awg, self.segments_bin)

	def add_awgs(self, awg):
		for i in awg:
			self.awg.add_awg(i)

	def mk_segment(self, name):
		return self.segments_bin.new(name)

	def get_segment(self, name):
		return self.segments_bin.get(name)

	def add_sequence(self,name,seq):
		self.sequencer.add_sequence(name, seq)

	def start_sequence(self, name):
		self.sequencer.start_sequence(name)
		
	def show_sequences(self):
		self.segments_bin.print_segments_info()

	def upload_data():

	    return
	def play():
		return

class segment_bin():

	def __init__(self, channels):
		self.segment = []
		self.channels = channels
		self.virtual_gate_matrix = None
		return	

	def new(self,name):
		if self.exists(name):
			raise ValueError("sement with the name : % \n alreadt exists"%name)
		self.segment.append(segment_container(name,self.channels))
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
	def __init__(self, awg_system, segment_bin):
		self.awg = awg_system
		self.segment_bin = segment_bin
		self.channels = segment_bin.channels
		self.sequences = dict()

	def add_sequence(self, name, sequence):
		self.sequences[name] = sequence

	def start_sequence(self, name):
		self.awg.upload(self.sequences[name], self.get_sequence_upload_data(name))
		self.awg.start(self.sequences[name])

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

		for i in self.channels:
			sequence_data_single_channel = []
			for k in seq:
				input_data = {'segment':k[0], 'ntimes':k[1]}
				unique = getattr(self.segment_bin.get_segment(k[0]), i).unique
				input_data['unique'] = unique
				# Make unique uuid's for each segment
				if unique == True:
					input_data['identifier'] = [uuid.uuid4() for i in range(k[1])]

				sequence_data_single_channel.append(input_data)

			upload_data[i] = sequence_data_single_channel

		return upload_data


p = pulselib()
seg  = p.mk_segment('INIT')
seg2 = p.mk_segment('Manip')
seg3 = p.mk_segment('Readout')

# append functions?
seg.P1.add_pulse([[10,0.5]
				 ,[20,0.5]])

seg.B0.add_pulse([[20,0],[30,0.5], [30,0]])
seg.B0.add_block(40,70,1)
seg.B0.add_pulse([[70,0],
				 [80,0],
				 [150,0.5],
				 [150,0]])
# seg.B0.repeat(20)
# seg.B0.wait(20)
# print(seg.B0.my_pulse_data)
# seg.reset_timevoltage_range_reset_needed()
seg.B1.add_pulse([[10,0],
				[10,1],
				[20,1],
				[20,0]])
seg.B1.add_block(20,50,1.4)

seg.B1.add_block(80,90,1.4)
seg.B1.wait(2000)
# seg.B1.plot_sequence()

seg2.B5.add_block(30,60,1)
seg3.B5.add_block(30,600,0.1)
seg3.B5.wait(2000)
p.show_sequences()

SEQ = [['INIT', 2, 0], ['Manip', 1, 0], ['INIT', 1, 0] ]

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