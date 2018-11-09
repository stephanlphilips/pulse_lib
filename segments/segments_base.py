import numpy as np
import datetime

from data_handling_functions import loop_controller, linspace, get_union_of_shapes, update_dimension
from data_classes import pulse_data
import copy

def last_edited(f):
	'''
	just a simpe decoter used to say that a certain wavefrom is updaded and therefore a new upload needs to be made to the awg.
	'''
	def wrapper(*args):
		args[0]._last_edit = datetime.datetime.now()
		return f(*args)
	return wrapper

class segment_single():
	'''
	Class defining single segments for one sequence.
	This is at the moment rather basic. Here should be added more fuctions.
	'''
	def __init__(self):
		self.type = 'default'
		self.to_swing = False

		# variable specifing the laetest change to the waveforms
		self._last_edit = datetime.datetime.now()
		
		# store data in numpy looking object for easy operator access.
		self.data = np.empty([1], dtype=object)
		self.data[0] = pulse_data()


		self.last = None
		self.unique = False

		# references to other channels (for virtual gates).
		self.reference_channels = []
		# reference channels for IQ virtual channels
		self.IQ_ref_channels = []
		# local copy of self that will be used to count up the virtual gates.
		self._pulse_data_all = None
		# variable specifing the lastest time the pulse_data_all is updated
		self.last_render = datetime.datetime.now() 

	def add_reference_channel(self, channel_name, segment_data, multiplication_factor):
		'''
		Add channel reference, this can be done to make by making a pointer to another segment.
		Args:
			Channels_name (str): human readable name of the virtual gate.
			segment_data (segment_single): pointer so the segment corresponsing the the channel name
			multiplication_factor (float64): times how much this segment should be added to the current one.
		'''
		virtual_segment = {'name': channel_name, 'segment': segment_data, 'multiplication_factor': multiplication_factor}
		self.reference_channels.append(virtual_segment)

	def add_IQ_ref_channel(self, channel_name, pointer_to_channel, I_or_Q_part):
		'''
		Add a reference to an IQ channel. Same principle as for the virtual one.
		Args:
			channel_name (str): human readable name of the virtual channel
			pointer_to_channel (*segment_single_IQ): pointer to segment_single_IQ object
			I_or_Q_part (str) : 'I' or 'Q' to indicate that the reference is to the I or Q part of the signal.
		'''
		virtual_IQ_segment = {'name': channel_name, 'segment': pointer_to_channel, 'I/Q': I_or_Q_part}
		self.IQ_ref_channels.append(virtual_IQ_segment)
		
	@loop_controller
	def reset_time(self):
		self.data_tmp.reset_time()

	@last_edited
	@loop_controller
	def add_pulse(self,array):
		'''
		Add manually a pulse.
		Args: 
			array (np.ndarray): array with times of the pulse.

		format array: 
		[[t0, Amp0],[t1, Amp1], ... [tn, Ampn]]
		'''
		if array[0] != [0.,0.]:
			array = [[0,0]] + array
		if self.data_tmp.start_time != 0:
			array = [[0,0]] + array

		arr = np.asarray(array)
		arr[1:,0] = arr[1:,0] + self.data_tmp.start_time 

		self.data_tmp.add_pulse_data(arr)


	@last_edited
	@loop_controller
	def add_block(self,start,stop, amplitude):
		'''
		add a block pulse on top of the existing pulse.
		'''
		if start != 0:
			pulse = np.array([[0,0], [start + self.data_tmp.start_time, 0], [start + self.data_tmp.start_time,amplitude], [stop + self.data_tmp.start_time, amplitude], [stop + self.data_tmp.start_time, 0]], dtype=np.double)
		else:
			pulse = np.array([[start + self.data_tmp.start_time, 0], [start + self.data_tmp.start_time,amplitude], [stop + self.data_tmp.start_time, amplitude], [stop + self.data_tmp.start_time, 0]], dtype=np.double)

		self.data_tmp.add_pulse_data(pulse)

	@last_edited
	@loop_controller
	def wait(self, wait):
		'''
		wait for x ns after the lastest wave element that was programmed.
		'''
		amp_0 = self.data_tmp.my_pulse_data[-1,1]
		t0 = self.data_tmp.total_time
		pulse = [[wait+t0, amp_0]]
		self.add_pulse(pulse)

	@last_edited
	@loop_controller
	def add_sin(self, start, stop, amp, freq, phase_offset=0):
		'''
		add a sinus to the current segment, parameters should be self exlenatory.
		The pulse will have a relative phase (as this is needed to all IQ work).
		'''
		self.data_tmp.add_sin_data(
			{'start_time' : start + self.data_tmp.start_time,
			'stop_time' : stop + self.data_tmp.start_time,
			'amplitude' : amp,
			'frequency' : freq,
			'phase_offset' : phase_offset})

	@last_edited
	@loop_controller
	def repeat(self, number):
		'''
		repeat a waveform n times.
		'''
		if number <= 1:
			return

		# if repeating elemenets with double points in the start/end, we don't want them in the segement, so we will strip the first and add them later (back in the big sequence).
		my_pulse_data_copy = copy.copy(self.data_tmp.my_pulse_data)
		if my_pulse_data_copy[-1,0] < self.data_tmp.total_time:
			my_pulse_data_copy = np.append(my_pulse_data_copy, [[self.data_tmp.total_time, my_pulse_data_copy[-1,1]]], axis=0)

		front_pulse_corr = None
		back_pulse_corr = None
		# check if there is twice the same starting number 
		
		if my_pulse_data_copy[0,0] == my_pulse_data_copy[1,0]:
			front_pulse_corr = my_pulse_data_copy[0]
			my_pulse_data_copy = my_pulse_data_copy[1:]

		if my_pulse_data_copy[-1,0] == my_pulse_data_copy[-2,0]:
			back_pulse_corr = my_pulse_data_copy[-1]
			my_pulse_data_copy = my_pulse_data_copy[:-1]


		pulse_data = np.zeros([my_pulse_data_copy.shape[0]*number, my_pulse_data_copy.shape[1]])

		sin_data = []
		total_time = self.data_tmp.total_time
		indices = 0

		for i in range(number):
			new_pulse = copy.copy(my_pulse_data_copy)
			
			new_pulse[:,0] +=  total_time*i
			pulse_data[indices:indices + new_pulse.shape[0]] = new_pulse
			indices += new_pulse.shape[0]

			for sin_data_item in self.data_tmp.sin_data:
				sin_data_item_new = copy.copy(sin_data_item)
				sin_data_item_new['start_time'] += total_time*i
				sin_data_item_new['stop_time'] += total_time*i
				sin_data.append(sin_data_item_new)

		if front_pulse_corr is not None:
			corr_pulse = np.zeros([pulse_data.shape[0] + 1, pulse_data.shape[1]])
			corr_pulse[1:] = pulse_data
			corr_pulse[0] = front_pulse_corr
			pulse_data = corr_pulse

		if back_pulse_corr is not None:
			corr_pulse = np.zeros([pulse_data.shape[0] + 1, pulse_data.shape[1]])
			corr_pulse[:-1] = pulse_data
			back_pulse_corr[0] = pulse_data[-1,0]
			corr_pulse[-1] = back_pulse_corr
			pulse_data = corr_pulse

		self.data_tmp.my_pulse_data = pulse_data
		self.data_tmp.sin_data = sin_data

	@last_edited
	@loop_controller
	def add_np(self,start, array):
		raise NotImplemented

	def __add__(self, other):
		'''
		define addition operator for segment_single
		'''
		new_segment = segment_single()
		if type(other) is segment_single:
			shape1 = self.data.shape
			shape2 = other.data.shape
			new_shape = get_union_of_shapes(shape1, shape2)
			other.data = update_dimension(other.data, new_shape)
			self.data= update_dimension(self.data, new_shape)
			new_segment.data = copy.copy(self.data)
			new_segment.data += other.data

		elif type(other) == int or type(other) == float:
			new_segment.data = copy.copy(self.data)
			new_segment.data += other
		else:
			raise TypeError("Please add up segment_single type or a number ")

		return new_segment

	def __sub__(self, other):
		raise NotImplemented

	def __mul__(self, other):
		'''
		muliplication operator for segment_single
		'''
		new_segment = segment_single()
		
		if type(other) is segment_single:
			shape1 = self.data.shape
			shape2 = other.data.shape
			new_shape = get_union_of_shapes(shape1, shape2)
			other.data = update_dimension(other.data, new_shape)
			self.data= update_dimension(self.data, new_shape)
			new_segment.data = copy.copy(self.data)
			new_segment.data *= other.data

		elif type(other) == int or type(other) == float:
			new_segment.data = copy.copy(self.data)
			new_segment.data *= other
		else:
			raise TypeError("Please add up segment_single type or a number ")

		return new_segment

	def __truediv__(self, other):
		raise NotImplemented



	@property
	def total_time(self,):
		return self.data.total_time


	def get_segment(self, points= None, t_start = 0, pre_delay = 0, post_delay = 0):
		'''
		input:
			Number of points of the raw sequence (None, if you just want to plot it. (without the delays))
			t_start: effective start time in the sequence, needed for unique segments  (phase coherent microwaves between segments)
			pre_delay : number of points to push before the sequence
			post delay: number of points to push after the sequence.

		Returns:
			A numpy array that contains the points for each ns
			points is the expected lenght.
		'''
		t, wvf = self._generate_segment(pre_delay, post_delay)
		return wvf

	@property
	def v_max(self):
		return self.pulse_data_all.get_vmax()

	@property
	def v_min(self):
		return self.pulse_data_all.get_vmin()

	@property
	def pulse_data_all(self):
		if self.last_edit > self.last_render or self._pulse_data_all is None:
			self._pulse_data_all = copy.copy(self)
			for ref_chan in self.reference_channels:
				self._pulse_data_all += ref_chan['segment']*ref_chan['multiplication_factor']

			self.last_render = self.last_edit
		return self._pulse_data_all

	@property
	def last_edit(self):
		for i in self.reference_channels:
			if self._last_edit < i['segment']._last_edit:
				self._last_edit = i['segment']._last_edit

		return self._last_edit
	
	def _generate_segment(self, pre_delay = 0, post_delay = 0):
		'''
		generate numpy array of the segment
		Args:
			pre_delay: predelay of the pulse (in ns) (e.g. for compensation of diffent coax length's)
			post_delay: extend the pulse for x ns
		'''


		pulse_data_all = self.pulse_data_all
		t_tot = pulse_data_all.total_time

		times = np.linspace(-pre_delay, int(t_tot-1 + post_delay), int(t_tot + pre_delay + post_delay))
		my_sequence = np.zeros([int(t_tot + pre_delay + post_delay)])

		for i in range(0,len(pulse_data_all.my_pulse_data)-1):
			t0 = int(pulse_data_all.my_pulse_data[i,0])
			t1 = int(pulse_data_all.my_pulse_data[i+1,0])
			my_sequence[t0 + pre_delay: t1 + pre_delay] = np.linspace(pulse_data_all.my_pulse_data[i,1], pulse_data_all.my_pulse_data[i+1,1], t1-t0)
		
		for sin_data_item in pulse_data_all.sin_data:
			start = int(round(sin_data_item['start_time'])) + pre_delay
			stop =  int(round(sin_data_item['stop_time'])) + pre_delay
			amp  =  sin_data_item['amplitude']
			freq =  sin_data_item['frequency']
			phase = sin_data_item['phase_offset']
			my_sequence[start:stop] += amp*np.sin(np.linspace(start, stop-1, stop-start)*freq*1e-9*2*np.pi + phase)

		return times, my_sequence

	def plot_segment(self):
		x,y = self._generate_segment()
		plt.plot(x,y)
		# plt.show()

			

# seg = segment_single()
# seg.add_block(0,linspace(2,20,18, axis=1),2)
# seg.wait(20)
# seg.reset_time()
# seg.add_sin(0,20,1,1e9,0)
# # seg.repeat(5)

# print(seg.data.shape)
# print(seg.data[0,0].my_pulse_data.T)
# # print(seg.data[0,0].sin_data)

# seg*=2 
# print(seg.data[0,0].sin_data)