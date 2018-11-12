import numpy as np
import datetime

from data_handling_functions import loop_controller, linspace, get_union_of_shapes, update_dimension
from data_classes import pulse_data

from segments_c_func import py_calc_value_point_in_between
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
			array (list): array with times of the pulse.

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
		Args:
			wait (double) : time in ns to wait
		'''
		amp_0 = self.data_tmp.my_pulse_data[-1,1]
		t0 = self.data_tmp.total_time

		if [t0, amp_0] != [0.,0.]:
			pulse = np.asarray([[0,0],[t0, amp_0],[wait+t0, amp_0]])
		else:
			pulse = np.asarray([[t0, amp_0],[wait+t0, amp_0]])

		self.data_tmp.add_pulse_data(pulse)

	@last_edited
	@loop_controller
	def add_sin(self, start, stop, amp, freq, phase_offset=0):
		'''
		add a sinus to the current segment, parameters should be self exlenatory.
		The pulse will have a relative phase (as this is needed to all IQ work).
		Args:
			start (double) : start time in ns of the pulse
			stop (double) : stop time in ns of the pulse
			amp (double) : amplitude of the pulse
			freq (double) : frequency of the pulse
			phase_offset (double) : offset in phase is needed
		'''
		self.data_tmp.add_sin_data(
			{'start_time' : start + self.data_tmp.start_time,
			'stop_time' : stop + self.data_tmp.start_time,
			'amplitude' : amp,
			'frequency' : freq,
			'phase' : phase_offset})

	@last_edited
	@loop_controller
	def repeat(self, number):
		'''
		repeat a waveform n times.
		Args:
			number (int) : number of ties to repeat the waveform
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


	def get_segment(self, index,total_time = None, pre_delay = 0, post_delay = 0, sample_rate=1e9):
		'''
		input:
			index of segment (list) : which segment to render (e.g. [0] if dimension is 1 or [2,5,10] if dimension is 3)
			total_time : total time of the segment in ns () (if not given the total time of the segment will be rendered)
			pre_delay : number of points to push before the sequence
			post delay: number of points to push after the sequence.
			sample rate : #/s (number of samples per second)

		Returns:
			A numpy array that contains the points for each ns
			points is the expected lenght.
		'''
		t, wvf = self._generate_segment(index, total_time, pre_delay, post_delay, sample_rate)
		return t, wvf

	@property
	def v_max(self):
		return self.pulse_data_all.get_vmax()

	@property
	def v_min(self):
		return self.pulse_data_all.get_vmin()

	@property
	def pulse_data_all(self):
		'''
		pulse data object that contains the counted op data of all the reference channels (e.g. IQ and virtual gates).
		'''
		if self.last_edit > self.last_render or self._pulse_data_all is None:
			self._pulse_data_all = copy.copy(self.data)
			for ref_chan in self.reference_channels:
				self._pulse_data_all += ref_chan['segment'].data*ref_chan['multiplication_factor']
			for ref_chan in self.IQ_ref_channels:
				self._pulse_data_all += ref_chan['segment'].get_IQ_data(ref_chan['I/Q'])

			self.last_render = self.last_edit

		return self._pulse_data_all

	@property
	def last_edit(self):
		for i in self.reference_channels:
			if self._last_edit < i['segment']._last_edit:
				self._last_edit = i['segment']._last_edit
		for i in self.IQ_ref_channels:
			if self._last_edit < i['segment']._last_edit:
				self._last_edit = i['segment']._last_edit

		return self._last_edit
	
	def _generate_segment(self, index, total_time=None, pre_delay = 0, post_delay = 0, sample_rate = 1e9):
		'''
		generate numpy array of the segment
		Args:
			pre_delay: predelay of the pulse (in ns) (e.g. for compensation of diffent coax length's)
			post_delay: extend the pulse for x ns
		'''

		# express in Gs/s
		sample_rate = sample_rate*1e-9
		sample_time_step = 1/sample_rate

		
		# get object with the concerning data
		flat_index = np.ravel_multi_index(tuple(index), self.pulse_data_all.shape)
		pulse_data_all_curr_seg = self.pulse_data_all.flat[flat_index]

		t_tot = total_time
		if total_time is None:
			t_tot = pulse_data_all_curr_seg.total_time

		t_tot_pt = get_effective_point_number(t_tot, sample_time_step)
		pre_delay_pt = get_effective_point_number(pre_delay, sample_time_step)
		post_delay_pt = get_effective_point_number(post_delay, sample_time_step)

		# render the time (maybe this should be removed...)
		times = np.linspace(-pre_delay_pt*sample_time_step, (t_tot_pt + post_delay_pt)*sample_time_step-sample_time_step, t_tot_pt + pre_delay_pt + post_delay_pt)
		my_sequence = np.zeros([t_tot_pt + pre_delay_pt + post_delay_pt])

		for i in range(0,len(pulse_data_all_curr_seg.my_pulse_data)-1):
			t0_pt = get_effective_point_number(pulse_data_all_curr_seg.my_pulse_data[i,0], sample_time_step)
			t1_pt = get_effective_point_number(pulse_data_all_curr_seg.my_pulse_data[i+1,0], sample_time_step)
			t0 = t0_pt*sample_time_step
			t1 = t1_pt*sample_time_step
			if t0 > t_tot:
				continue
			elif t1 > t_tot:
				val = py_calc_value_point_in_between(pulse_data_all_curr_seg.my_pulse_data[i,:], pulse_data_all_curr_seg.my_pulse_data[i,:], t_tot)
				my_sequence[t0_pt + pre_delay_pt: t_tot_pt + pre_delay_pt] = np.linspace(
					pulse_data_all_curr_seg.my_pulse_data[i,1], 
					val, t_tot_pt-t0_pt)
			else:
				my_sequence[t0_pt + pre_delay_pt: t1_pt + pre_delay_pt] = np.linspace(pulse_data_all_curr_seg.my_pulse_data[i,1], pulse_data_all_curr_seg.my_pulse_data[i+1,1], t1_pt-t0_pt)
		
		for sin_data_item in pulse_data_all_curr_seg.sin_data:
			if sin_data_item['start_time'] > t_tot:
				continue
			elif sin_data_item['stop_time'] > t_tot:
				stop = t_tot_pt + pre_delay_pt
			else:
				stop =  get_effective_point_number(sin_data_item['stop_time'], sample_time_step) + pre_delay_pt
			
			start = get_effective_point_number(sin_data_item['start_time'], sample_time_step) + pre_delay_pt
			start_t  = (start - pre_delay_pt)*sample_time_step
			stop_t  = (stop - pre_delay_pt)*sample_time_step

			amp  =  sin_data_item['amplitude']
			freq =  sin_data_item['frequency']
			phase = sin_data_item['phase']
			print("redering freq = ",freq*1e-9*2*np.pi, phase)
			print(np.linspace(start_t, stop_t-sample_time_step, stop-start)*freq*1e-9*2*np.pi + phase)
			my_sequence[start:stop] += amp*np.sin(np.linspace(start_t, stop_t-sample_time_step, stop-start)*freq*1e-9*2*np.pi + phase)
			print(amp*np.sin(np.linspace(start_t, stop_t-sample_time_step, stop-start)*freq*1e-9*2*np.pi + phase))
		return times, my_sequence

	def plot_segment(self):
		x,y = self._generate_segment()
		plt.plot(x,y)
		# plt.show()

def get_effective_point_number(time, time_step):
	'''
	function that discretizes time depending on the sample rate of the AWG.
	Args:
		time (double): time in ns of which you want to know how many points the AWG needs to get there
		time_step (double) : time step of the AWG (ns)

	Returns:
		how many points you need to get to the desired time step.
	'''
	n_pt, mod = divmod(time, time_step)
	if mod > time_step/2:
		n_pt += 1

	return int(n_pt)
