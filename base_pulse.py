import numpy as np
import matplotlib.pyplot as plt 

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
		self.awg_channels = ['P1','P2','P3','P4','P5','B0','B1','B2','B3','B4','B5',]
		self.awg_channels_to_physical_locations = ['here_dict']
		self.awg_channels_kind = []
		self.awg_markers =['mkr1', 'mkr2', 'mkr3' ]
		self.awg_markers_to_location = []
		self.delays = []
		self.convertion_matrix= []
		self.segments_bin = segment_bin(self.awg_channels)

		self.devices = []

		self.sequencer =  sequencer(self.awg_channels, self.awg_markers)
		# Keysight properties.
		self.backend = 'keysight'
		self.frequency = 1e9
		self.max_mem = 2e9
		self.mem_used = 0
		self.in_mem = []

	def add_awgs(self, awg):
		for i in awg:
			self.devices.append(i)

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

	def reset_time(self):
		maxtime = 0
		for i in self.channels:
			k = getattr(self, i)
			t = k.get_total_time()
			if t > maxtime:
				maxtime = t
		for i in self.channels:
			getattr(self, i).starttime = maxtime

class segment_single():
	def __init__(self):
		self.type = 'default'
		self.to_swing = False
		self.starttime = 0

		self.my_pulse_data = np.zeros([1,2])
		self.last = None
		self.IQ_data = [] #todo later.

	def add_pulse(self,array):
		'''
		format :: 
		[[t0, Amp0],[t1, Amp1]]
		'''
		arr = np.asarray(array)
		arr[:,0] = arr[:,0] + self.starttime 

		self.my_pulse_data = np.append(self.my_pulse_data,arr, axis=0)

	def add_block(self,start,stop, amplitude):
		amp_0 = self.my_pulse_data[-1,1]
		print(amp_0)
		pulse = [ [start, amp_0], [start,amplitude], [stop, amplitude], [stop, amp_0]]
		self.add_pulse(pulse)

	def wait(self, wait):
		amp_0 = self.my_pulse_data[-1,1]
		t0 = self.my_pulse_data[-1,0]
		pulse = [wait+t0, amp_0]

	def repeat(self, number):
		return

	def add_np(self,start, array):
		return

	def add_IQ_pair():
		return

	def add():
		return
	
	def get_total_time(self):
		return self.my_pulse_data[-1,0]

	def get_sequence(self, voltage_range = None, off_set = None):
		'''
		Returns a numpy array that contains the points for each ns
		'''
		return None

	def _generate_sequence(self):
		# 1 make base sequence:
		t_tot = self.my_pulse_data[-1,0]

		times = np.linspace(0, int(t_tot-1), int(t_tot))
		my_sequence = np.empty([int(t_tot)])


		for i in range(0,len(self.my_pulse_data)-1):
			my_loc = np.where(times < self.my_pulse_data[i+1,0])[0]
			my_loc = np.where(times[my_loc] >= self.my_pulse_data[i,0])[0]

			if my_loc.size==0:
				continue;

			end_voltage = self.my_pulse_data[i,1] + (self.my_pulse_data[i+1,1] - self.my_pulse_data[i,1])*(times[my_loc[-1]] - self.my_pulse_data[i,0])/(self.my_pulse_data[i+1,0]- self.my_pulse_data[i,0]);

			start_stop_values = np.linspace(self.my_pulse_data[i,1],end_voltage,my_loc.size);
			my_sequence[my_loc] = start_stop_values;

		return times, my_sequence

	def plot_sequence(self):
		x,y = self._generate_sequence()
		plt.plot(x,y)
		plt.show()
		
class marker_single():
	def __init__(self):
		self.type = 'default'
		self.swing = False
		self.latest_time = 0

		self.my_pulse_data = np.zeros([1,2])

	def add(self, start, stop):
		self.my_pulse_data = np,append(self.my_pulse_data, [[start,0],[start,1],[stop,1],[stop,0]])

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

	def upload(self, name):
		time = self.get_max_time(name)
		upload_data = np.array[len(channels), time]
		for i in range(len(channels)):
			upload_data[i] = self.segment.

class sequencer():
	def __init__(self, channels, markers, segment_bin):
		self.channels = channels
		self.markers  = markers
		self.segment_bin = segment_bin
		self.sequences = dict()

	def add_sequence(self, name, sequence):
		self.sequences['name'] = sequence

	def start_sequence(self, name):
		# Ask segmentbin to check if elements are present, if not -- upload
		self.segment_bin.upload()
		# Upload the relevant segments.

class keysight_awg():
	def __init__(awg_object):
		self.awg = awg_object
		self.usable_mem = 1e9
		self.current_waveform_number = 0

	def upload_waveform(self, wfv):
		if self.usable_mem - len(wfv) < 0:
			raise Exception("AWG Full :(. Clear all the ram... Note that this error should normally be prevented automatically.")

		# wfv.astype(np.int16)
	def clear_mem(self):
		return

	def check_mem_availability(self, num_pt):
		return True



p = pulselib()
seg = p.mk_segment('INIT')
seg2 = p.mk_segment('Manip')
seg3 = p.mk_segment('Readout')

# append functions?
seg.B0.add_pulse([[10,5]
				 ,[20,5]])

seg.B0.add_pulse([[20,0],[30,5], [30,0]])
seg.B0.add_block(40,70,2)
seg.B0.add_pulse([[70,0],
				 [80,0],
				 [150,5],
				 [150,0]])
# seg.B0.repeat(20)
# seg.B0.wait(20)
# print(seg.B0.my_pulse_data)
# seg.reset_time()
seg.B1.add_pulse([[10,0],
				[10,5],
				[20,5],
				[20,0]])
seg.B1.add_block(20,50,2)

seg.B1.add_block(80,90,2)
# seg.B1.plot_sequence()
p.show_sequences()

SEQ = [['INIT', 1, 0], ['Manip', 1, 0], ['Readout', 1, 0] ]

p.add_sequence('mysequence', SEQ)

p.start_sequence('mysequence')

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

