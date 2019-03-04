import numpy as np
import datetime

from pulse_lib.segments.data_handling_functions import loop_controller, get_union_of_shapes, update_dimension
from pulse_lib.segments.data_classes import pulse_data, data_container, get_effective_point_number
from pulse_lib.segments.looping import loop_obj
import copy

import matplotlib.pyplot as plt

def last_edited(f):
	'''
	just a simpe decoter used to say that a certain wavefrom is updaded and therefore a new upload needs to be made to the awg.
	'''
	def wrapper(*args):
		if args[0].render_mode == True:
			ValueError("cannot alter this segment, this segment ({}) in render mode.".format(args[0].name))
		args[0]._last_edit = datetime.datetime.now()
		return f(*args)
	return wrapper

class segment_single():
	'''
	Class defining single segments for one sequence.
	This is at the moment rather basic. Here should be added more fuctions.
	'''
	def __init__(self, name, segment_type = 'render'):
		'''
		Args:
			name (str): name of the segment usually the channel name
			segment_type (str) : type of the segment (e.g. 'render' --> to be rendered, 'virtual'--> no not render.)
		'''
		self.type = segment_type
		self.name = name
		self.render_mode = False
		# variable specifing the laetest change to the waveforms
		self._last_edit = datetime.datetime.now()
		
		# store data in numpy looking object for easy operator access.
		self.data = data_container(pulse_data())

		self.last = None
		self.unique = False

		# references to other channels (for virtual gates).
		self.reference_channels = []
		# reference channels for IQ virtual channels
		self.IQ_ref_channels = []
		# local copy of self that will be used to count up the virtual gates.
		self._pulse_data_all = None
		# data caching variable. Used for looping and so on (with a decorator approach)
		self.data_tmp = None
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
	
	@last_edited
	@loop_controller
	
	def reset_time(self, time=None, extend_only = False):
		'''
		resets the time back to zero after a certain point
		Args: 
			time (double) : (optional), after time to reset back to 0. Note that this is absolute time and not rescaled time.
		'''
		self.data_tmp.reset_time(time, extend_only)

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
	def add_ramp(self, start, stop, amplitude, keep_amplitude=False):
		'''
		Makes a linear ramp
		Args:
			start (double) : starting time of the ramp
			stop (double) : stop time of the ramp
			amplitude : total hight of the ramp, starting from the base point
			keep_amplitude : when pulse is done, keep reached amplitude for time infinity
		'''
		if keep_amplitude == False:
			if start != 0:
				pulse = np.array([[0,0], [start + self.data_tmp.start_time, 0], [stop + self.data_tmp.start_time, amplitude], [stop + self.data_tmp.start_time, 0]], dtype=np.double)
			else:
				pulse = np.array([[start + self.data_tmp.start_time, 0],  [stop + self.data_tmp.start_time, amplitude], [stop + self.data_tmp.start_time, 0]], dtype=np.double)
		else:
			if start != 0:
				pulse = np.array([[0,0], [start + self.data_tmp.start_time, 0], [stop + self.data_tmp.start_time, amplitude] ], dtype=np.double)
			else:
				pulse = np.array([[start + self.data_tmp.start_time, 0],  [stop + self.data_tmp.start_time, amplitude] ], dtype=np.double)

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

		pulse = np.asarray([[t0, 0],[wait+t0, 0]], dtype=np.double)

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
			{
			'type' : 'std', 
			'start_time' : start + self.data_tmp.start_time,
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

	@property
	def shape(self):
		return self.data.shape
	
	def __add__(self, other):
		'''
		define addition operator for segment_single
		'''
		new_segment = segment_single(self.name)
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
		return self.__add__(other*-1)

	def __mul__(self, other):
		'''
		muliplication operator for segment_single
		'''
		new_segment = segment_single(self.name)
		
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

	def __getitem__(self, *key):
		'''
		get slice or single item of this segment (note no copying, just referencing)
		Args:
			*key (int/slice object) : key of the element -- just use numpy style accessing (slicing supported)
		'''
		item = segment_single(self.name)
		item.type = self.type

		item.render_mode = self.render_mode 
		item._last_edit = self._last_edit
		item.data = self.data[key[0]]

		item.reference_channels = self.reference_channels 
		item.IQ_ref_channels = self.IQ_ref_channels

		return item

	@last_edited
	def append(self, other, time = None):
		'''
		Put the other segment behind this one.
		Args:
			other (segment_single) : the segment to be appended
			time (double/loop_obj) : attach at the given time (if None, append at total_time of the segment)

		A time reset will be done after the other segment is added.
		TODO: transfer of units
		'''
		other_loopobj = loop_obj()
		other_loopobj.add_data(other.data, axis=list(range(other.data.ndim -1,-1,-1)))

		self.__append(other_loopobj, time)

		return self

	@loop_controller
	def __append(self, other, time):
		"""
		Put the other segment behind this one (for single segment data object)
		Args:
			other (segment_single) : the segment to be appended
			time (double/loop_obj) : attach at the given time (if None, append at total_time of the segment)
		"""
		if time is None:
			time = self.data_tmp.total_time

		self.data_tmp.append(other, time)

	@last_edited
	@loop_controller
	def slice_time(self, start_time, stop_time):
		"""
		Cuts parts out of a segment.
		Args:
			start_time (double) : effective new start time
			stop_time (double) : new ending time of the segment

		The slice_time function allows you to cut a waveform in different sizes.
		This function should be handy for debugging, example usage would be, 
		You are runnning an algorithm and want to check what the measurement outcomes are though the whole algorithm.
		Pratically, you want to know
			0 -> 10ns (@10 ns still everything as expected?)
			0 -> 20ns
			0 -> ...
		This function would allow you to do that, e.g. by calling segment.cut_segment(0, lp.linspace(10,100,9))
		"""
		self.data_tmp.slice_time(start_time, stop_time)

	@property
	def total_time(self,):
		return self.data.total_time

	def get_segment(self, index, pre_delay = 0, post_delay = 0, sample_rate=1e9):
		'''
		input:
			index of segment (list) : which segment to render (e.g. [0] if dimension is 1 or [2,5,10] if dimension is 3)
			pre_delay : number of points to push before the sequence
			post delay: number of points to push after the sequence.
			sample rate : #/s (number of samples per second)

		Returns:
			A numpy array that contains the points for each ns
			points is the expected lenght.
		'''
		wvf = self._generate_segment(index, pre_delay, post_delay, sample_rate)
		return wvf

	def v_max(self, index, sample_rate = 1e9):
		index = np.ravel_multi_index(tuple(index), self.pulse_data_all.shape)

		return self.pulse_data_all.flat[index].get_vmax()

	def v_min(self, index, sample_rate = 1e9):
		index = np.ravel_multi_index(tuple(index), self.pulse_data_all.shape)
		return self.pulse_data_all.flat[index].get_vmin()

	def integrate(self, index, pre_delay = 0, post_delay = 0, sample_rate = 1e9):
		'''
		get integral value of the waveform (e.g. to calculate an automatic compensation)
		Args:
			index (tuple) : index of the concerning waveform
			pre_delay (double) : ns to delay before the pulse
			post_delay (double) : ns to delay after the pulse
			sample_rate (double) : rate at which to render the pulse
		returns: intergral value
		'''
		flat_index = np.ravel_multi_index(tuple(index), self.pulse_data_all.shape)
		pulse_data_all_curr_seg = self.pulse_data_all.flat[flat_index]

		total_time = pulse_data_all_curr_seg.total_time
		return pulse_data_all_curr_seg.integrate_waveform(pre_delay, total_time + post_delay, sample_rate)

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
	
	def _generate_segment(self, index, pre_delay = 0, post_delay = 0, sample_rate = 1e9):
		'''
		generate numpy array of the segment
		Args:
			index (tuple) : index of the segement to generate
			pre_delay: predelay of the pulse (in ns) (e.g. for compensation of diffent coax length's)
			post_delay: extend the pulse for x ns
			sample rate (double) : sample rate of the pulse to be rendered at.
		'''

		# get object with the concerning data
		flat_index = np.ravel_multi_index(tuple(index), self.pulse_data_all.shape)
		pulse_data_all_curr_seg = self.pulse_data_all.flat[flat_index]

		total_time = pulse_data_all_curr_seg.total_time

		my_sequence = pulse_data_all_curr_seg.render(pre_delay, post_delay, sample_rate)
		return my_sequence

	def plot_segment(self, index = [0], render_full = True):
		'''
		Args:
			index : index of which segment to plot
			render full (bool) : do full render (e.g. also get data form virtual channels). Put True if you want to see the waveshape send to the AWG.
		'''
		# standard 1 Gs/s
		sample_rate = 1e9
		sample_time_step = 1/sample_rate

		if render_full == True:
			flat_index = np.ravel_multi_index(tuple(index), self.pulse_data_all.shape)
			pulse_data_curr_seg = self.pulse_data_all.flat[flat_index]
		else:
			flat_index = np.ravel_multi_index(tuple(index), self.data.shape)
			pulse_data_curr_seg = self.data.flat[flat_index]

		y = pulse_data_curr_seg.render(0, 0, sample_rate)
		x = np.linspace(0, pulse_data_curr_seg.total_time*sample_time_step-sample_time_step, len(y))*1e9

		plt.plot(x,y, label=self.name)
		# plt.show()

