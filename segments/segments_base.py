import numpy as np
import datetime

import data_handling_functions as DHF
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
		self.starttime = 0
		# variable specifing the laetest change to the waveforms
		self._last_edit = datetime.datetime.now()
		
		# store data in numpy looking object for easy operator access.
		self.data = np.empty([1], dtype=object)
		self.data[0] = DHF.pulse_data()

		self.last = None
		self.unique = False

		# references to other channels (for virtual gates).
		self.reference_channels = []
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

	@last_edited
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
		if self.starttime != 0:
			array = [[0,0]] + array

		arr = np.asarray(array)
		arr[1:,0] = arr[1:,0] + self.starttime 

		self.data.add_pulse_data(arr)


	@last_edited
	def add_block(self,start,stop, amplitude):
		'''
		add a block pulse on top of the existing pulse.
		'''
		if start != 0:
			pulse = np.array([[0,0], [start + self.starttime, 0], [start + self.starttime,amplitude], [stop + self.starttime, amplitude], [stop + self.starttime, 0]], dtype=np.double)
		else:
			pulse = np.array([[start + self.starttime, 0], [start + self.starttime,amplitude], [stop + self.starttime, amplitude], [stop + self.starttime, 0]], dtype=np.double)
		self.data.add_pulse_data(pulse)

	@last_edited
	def wait(self, wait):
		'''
		wait for x ns after the lastest wave element that was programmed.
		'''
		amp_0 = self.data.my_pulse_data[-1,1]
		t0 = self.data.total_time
		pulse = [[wait+t0, amp_0]]
		self.add_pulse(pulse)

	@last_edited
	def add_sin(self, start, stop, amp, freq, phase_offset=0):
		'''
		add a sinus to the current segment, parameters should be self exlenatory.
		The pulse will have a relative phase (as this is needed to all IQ work).
		'''
		self.data.add_sin_data(
			{'start_time' : start + self.starttime,
			'stop_time' : stop + self.starttime,
			'amplitude' : amp,
			'frequency' : freq,
			'phase_offset' : phase_offset})

	@last_edited
	def repeat(self, number):
		'''
		repeat a waveform n times.
		'''
		if number <= 1:
			return

		# if repeating elemenets with double points in the start/end, we don't want them in the segement, so we will strip the first and add them later (back in the big sequence).
		my_pulse_data_copy = copy.copy(self.data.my_pulse_data)
		if my_pulse_data_copy[-1,0] < self.total_time:
			my_pulse_data_copy = np.append(my_pulse_data_copy, [[self.total_time, my_pulse_data_copy[-1,1]]], axis=0)

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
		total_time = self.total_time
		indices = 0

		for i in range(number):
			new_pulse = copy.copy(my_pulse_data_copy)
			
			new_pulse[:,0] +=  total_time*i
			pulse_data[indices:indices + new_pulse.shape[0]] = new_pulse
			indices += new_pulse.shape[0]

			for sin_data_item in self.data.sin_data:
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

		self.data.my_pulse_data = pulse_data
		self.data.sin_data = sin_data

	@last_edited
	def add_np(self,start, array):
		raise NotImplemented

	def __add__(self, other):
		'''
		define addition operator for segment_single
		'''
		new_segment = segment_single()
		if type(other) is segment_single:
			if len(other.my_pulse_data) == 1:
				new_segment.my_pulse_data = copy.copy(self.my_pulse_data)
			elif len(self.my_pulse_data) == 1:
				new_segment.my_pulse_data = copy.copy(other.my_pulse_data)
			else:
				new_segment.my_pulse_data = self._add_up_pulse_data(other.my_pulse_data)

			sin_data = copy.copy(self.sin_data)
			sin_data.extend(other.sin_data)
			new_segment.sin_data = sin_data
		elif type(other) == int or type(other) == float:
			new_pulse = copy.copy(self.my_pulse_data)
			new_pulse[:,1] += other
			new_segment.my_pulse_data = new_pulse
			new_segment.sin_data = self.sin_data

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
			raise NotImplemented
		elif type(other) == int or type(other) == float or type(other) == np.float64:
			new_pulse = copy.copy(self.my_pulse_data)
			new_pulse[:,1] *= other
			new_segment.my_pulse_data = new_pulse
			sin_data = []
			for i in self.sin_data:
				my_sin = copy.copy(i)
				my_sin['amplitude'] *=other
				sin_data.append(my_sin)

			new_segment.sin_data = sin_data
		else:
			raise TypeError("Please add up segment_single type or a number ")
		
		return new_segment

	def __truediv__(self, other):
		raise NotImplemented

	def _add_up_pulse_data(self, new_pulse):
		'''
		add a pulse up to the current pulse in the memory.
		new_pulse --> default format as in the add_pulse function
		'''
		my_pulse_data_copy = self.my_pulse_data
		# step 1: make sure both pulses have the same length
		if self.total_time < new_pulse[-1,0]:
			to_insert = [[new_pulse[-1,0],my_pulse_data_copy[-1,1]]]
			my_pulse_data_copy = self._insert_arrays(my_pulse_data_copy, to_insert, len(my_pulse_data_copy)-1)
		elif self.total_time > new_pulse[-1,0]:
			to_insert = [[my_pulse_data_copy[-1,0],new_pulse[-1,1]]]
			new_pulse = self._insert_arrays(new_pulse, to_insert, len(new_pulse)-1)
		
		my_pulse_data_tmp, new_pulse_tmp = seg_func.interpolate_pulses(my_pulse_data_copy, new_pulse)

		final_pulse = np.zeros([len(my_pulse_data_tmp),2])
		final_pulse[:,0] = my_pulse_data_tmp[:,0]
		final_pulse[:,1] +=  my_pulse_data_tmp[:,1]  + new_pulse_tmp[:,1]

		return final_pulse

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

	@staticmethod
	def _insert_arrays(src_array, to_insert, insert_position):
		'''
		insert pulse points in array
		Args:
			src_array : 2D pulse table
			to_insert : 2D pulse table to be inserted in the source
			insert_position: after which point the insertion needs to happen
		'''

		# calcute how long the piece is you want to insert
		dim_insert = len(to_insert)
		insert_position += 1

		new_arr = np.zeros([src_array.shape[0]+dim_insert, src_array.shape[1]])
		
		new_arr[:insert_position, :] = src_array[:insert_position, :]
		new_arr[insert_position:(insert_position + dim_insert), :] = to_insert
		new_arr[(insert_position + dim_insert):] = src_array[insert_position :]

		return new_arr


			

# seg = segment_single()
# seg.add_block(0,10,2)
# seg.wait(20)
# seg.add_sin(0,20,1,1e9,0)
# seg.repeat(5)

# print(seg.data.my_pulse_data)
# print(seg.data.sin_data)