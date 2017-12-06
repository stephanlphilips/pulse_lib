import numpy as np


class pulselib:
	'''
		Global class that is an organisational element in making the pulses.
		The idea is that you first make individula segments,
		you can than later combine them into a sequence, this sequence will be uploaded
	'''
	

	def __init__(self):
		self.awg_channels = ['ch1']
		self.awg_channels_to_physical_locations = ['here_dict']
		self.awg_channels_kind = []
		self.marker_channels =['names']
		self.marger_channels_to_location = []
		self.delays = []
		self.convertion_matrix= []
		self.segments_bin = segment_bin(self.awg_channels)

		self.backend = 'keysight'
		self.frequency = 1e9
	def mk_segment(self, name):
		return self.segments_bin.new(name)

	def get_segment(self, name):
		return self.segments_bin.get(name)
	def mk_sequence():
		return
	def upload_data():
	    return
	def play():
		return


class segment_container():
	'''
    Class containing all the single segments for for a series of channels.
	This is a organisational class.
    '''
	def __init__(self, name, channels):
		self.channels = channels
		self.name = name

		for i in self.channels:
			setattr(self, i, segment_single())


class segment_single():
	def __init__(self):
		self.test = 'test'

	def add_pulse(array, channel):
		return
	def add_IQ_pair():
		return

	def add():
		return
	def reset_time():
	    # aligns all time together -- the channel with the longest time will be chosen
	    return

class segment_bin():

	def __init__(self, channels):
		self.segment = []
		self.channels = channels
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

	def exists(self, name):
		for i in self.segment:
			if i.name ==name:
				return True
		return False
	
p = pulselib()
seg = p.mk_segment('test')
print(seg.ch1.test)
# class channel_data_obj():
#     #object containing the data for a specific channels
#     #the idea is that all the data is parmeterised and will be constuceted whenever the function is called.

#     self.my_data_array = np.empty()
    
#     add_data

# class block_pulses:
#     # class to make block pulses


# how to do pulses
# -> sin?
# -> pulses?
# -> step_pulses

# p = pulselin()

# seg = pulselib.mk_segment('manip')
# seg.p1.add_pulse(10,50, 20, prescaler= '1')
# seg.p3.add_pulse(12,34, 40,)
# seg.k2.add_pulse_advanced([pulse sequence])
# seg.add_np(array, tstart_t_stop
# seg.p5.add_sin(14,89, freq, phase, amp)

# pulse
