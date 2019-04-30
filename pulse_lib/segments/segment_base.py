"""
File containing the parent class where all segment objects are derived from.
"""

import numpy as np
from dataclasses import dataclass
import datetime

from pulse_lib.segments.utility.data_handling_functions import loop_controller, get_union_of_shapes, update_dimension
from pulse_lib.segments.data_classes.data_generic import data_container
from pulse_lib.segments.utility.looping import loop_obj

import copy

import matplotlib.pyplot as plt


@dataclass
class IQ_render_info:
	"""
	structure to save relevant information about the rendering of the IQ channels (for is in channel object).
	"""
	LO : float
	virtual_channel_name: str
	virtual_channel_pointer: loop_obj #TODO fix to segment_IQ data type, needs to be post loaded somehow. 
	IQ_render_option : str
	image_render_option : str


def last_edited(f):
	'''
	just a simpe decorater used to say that a certain wavefrom is updaded and therefore a new upload needs to be made to the awg.
	'''
	def wrapper(*args):
		if args[0].render_mode == True:
			ValueError("cannot alter this segment, this segment ({}) in render mode.".format(args[0].name))
		args[0]._last_edit = datetime.datetime.now()
		return f(*args)
	return wrapper

class segment_base():
	'''
	Class defining base function of a segment. All segment types should support all operators.
	If you make new data type, here you should buil-in in basic support to allow for general operations.
			
	For an example, look in the data classes files.
	'''
	def __init__(self, name, data_object, segment_type = 'render'):
		'''
		Args:
			name (str): name of the segment usually the channel name
			data_object (object) : class that is used for saving the data type.
			segment_type (str) : type of the segment (e.g. 'render' --> to be rendered, 'virtual'--> no not render.)
		'''
		self.type = segment_type
		self.name = name
		self.render_mode = False
		# variable specifing the laetest change to the waveforms
		self._last_edit = datetime.datetime.now()
		
		# store data in numpy looking object for easy operator access.
		self.data = data_container(data_object)

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

		self.loops = list()
		# some basic properties.
		self._names = tuple()
		self._setvals = tuple()
		self._units = tuple()

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

	def add_IQ_channel(self, LO, channel_name, pointer_to_channel, I_or_Q_part, image):
		'''
		Add a reference to an IQ channel. Same principle as for the virtual one.
		Args:
			LO (float) : frequecy at which MW source runs (needed to calculate final IQ signal.)
			channel_name (str): human readable name of the virtual channel
			pointer_to_channel (*segment_single_IQ): pointer to segment_single_IQ object
			I_or_Q_part (str) : 'I' or 'Q' to indicate that the reference is to the I or Q part of the signal.
			image (str) : '+' / '-', take the image of the signal (needed for differential inputs)
		'''
		self.IQ_ref_channels.append(IQ_render_info(LO, channel_name, pointer_to_channel, I_or_Q_part, image))
	
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
	def wait(self, time):
		'''
		resets the time back to zero after a certain point
		Args: 
			time (double) : time in ns to wait
		'''
		self.data_tmp.wait(time)

	@property
	def shape(self):
		return self.data.shape

	@property
	def ndim(self):
		return self.data.ndim

	@property
	def names(self):
		self.__generate_properties()
		return self._names
	
	@property
	def units(self):
		self.__generate_properties()
		return self._units

	@property
	def setvals(self):
		self.__generate_properties()
		return self._setvals

	def __generate_properties(self):
		'''
		generate properties like names, units and setvals of the current segment.

		Note that this are the units of this segment and not the sements around it.
		Note2 : for a loopobject with a dimensions bigger than one, the setvals are not settable, as a 1D array for each of them is expected. As such, only the first row of the array will be kept... (though it is unlikely that this scenario will occur).
		'''
		
		
		units = ["a.u."]*self.ndim
		names = ["undefined"]*self.ndim
		setvals = []*self.ndim
		enforce_setvals = [False]*self.ndim

		for lp in range(self.loops):
			for i in range(lp.ndim):
				if lp.units[i] != 'a.u.':
					units[lp.axis[i]] = lp.units[i]
				if lp.names[i] != 'undefined':
					names[lp.axis[i]] = lp.names[i]
				if enforce_setvals[i] == False:
					locations = [0] * lp.ndim
					locations[i] = slice(0, None)

					setvals[lp.axis[i]] = lp.setvals[ tuple(locations) ]
					enforce_setvals[lp.axis[i]] = lp.setvals_set

		self._names = names
		self._units = units
		self._setvals = setvals

	def __add__(self, other):
		'''
		define addition operator for segment_single
		'''
		new_segment = copy.copy(self)
		if isinstance(other, segment_base):
			shape1 = new_segment.data.shape
			shape2 = other.data.shape
			new_shape = get_union_of_shapes(shape1, shape2)
			other.data = update_dimension(other.data, new_shape)
			new_segment.data= update_dimension(new_segment.data, new_shape)
			new_segment.data += other.data

		elif type(other) == int or type(other) == float:
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
		new_segment = copy.copy(self)
		
		if isinstance(other, segment_base):
			raise TypeError("muliplication of two segments not supported. Please multiply by a number.")
		elif type(other) == int or type(other) == float or type(other) == np.double:
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

	@last_edited
	@loop_controller
	def repeat(self, number):
		'''
		repeat a waveform n times.
		Args:
			number (int) : number of ties to repeat the waveform
		'''
		
		data_copy = copy.copy(self.data_tmp)
		for i in range(number-1):
			self.data_tmp.append(data_copy)
			
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

	def plot_segment(self, index = [0], render_full = True, sample_rate=1e9):
		'''
		Args:
			index : index of which segment to plot
			render full (bool) : do full render (e.g. also get data form virtual channels). Put True if you want to see the waveshape send to the AWG.
			sample_rate (float): standard 1 Gs/s
		'''
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


if __name__ == '__main__':
	s = segment_single("test")
	from pulse_lib.segments.looping import linspace
	s.add_block(0, 10, linspace(2,10,50, unit=("ee")))
	print(s.loops)
	print(s.units)

