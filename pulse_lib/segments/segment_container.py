"""
File contains an object that mananges segments. E.g. you are dealing with mutiple channels. 
This object also allows you to do operations on all segments at the same time.
"""

from pulse_lib.segments.segment_pulse import segment_pulse
from pulse_lib.segments.segment_IQ import segment_IQ
from pulse_lib.segments.segment_markers import segment_marker

import pulse_lib.segments.utility.looping as lp
from pulse_lib.segments.utility.data_handling_functions import find_common_dimension, update_dimension

import uuid

import numpy as np
import datetime


class segment_container():
	'''
	Class containing all the single segments for for a series of channels.
	This is a organisational class.
	Class is capable of checking wheather upload is needed.
	Class is capable of termining what volatages are required for each channel.
	Class returns vmin/vmax data to awg object
	Class returns upload data as a numpy<double> array to awg object.
	'''
	def __init__(self, channel_names, markers = [], virtual_gates_objs = [], IQ_channels_objs = []):
		"""
		initialize a container for segments.
		Args:
			channel_names (list<str>) : list with names of physical output channels on the AWG
			markers (list<str>) : declaration which of these channels are markers
			virtual_gates_objs (list<virtual_gates_constructor>) : list of object that define virtual gates
			IQ_channels_objs (list<IQ_channel_constructor>) : list of objects taht define virtual IQ channels.
		"""
		# physical channels
		self.r_channels = []
		# physical + virtual channels
		self.channels = []
		self.render_mode = False
		self.id = uuid.uuid4()
		self._Vmin_max_data = dict()

		for name in channel_names:
			self._Vmin_max_data[name] = {"v_min" : None, "v_max" : None}
		
		self.prev_upload = datetime.datetime.utcfromtimestamp(0)

		
		# define real channels (+ markers)
		for name in channel_names:
			if name in markers:
				setattr(self, name, segment_marker(name))
			else:
				setattr(self, name, segment_pulse(name))
			self.channels.append(name)
			self.r_channels.append(name)

		# define virtual gates
		for virtual_gates in virtual_gates_objs:
			# make segments for virtual gates.
			for virtual_gate_name in virtual_gates.virtual_gate_names:
				setattr(self, virtual_gate_name, segment_pulse(virtual_gate_name, 'virtual_baseband'))
				self.channels.append(virtual_gate_name)

			# add reference in real gates.
			for i in range(virtual_gates.size):
				real_channel = getattr(self, virtual_gates.virtual_gate_names[i])
				virtual_gates_values = self.virtual_gate_matrix[i,:]

				for j in range(virtual_gates.size):
					if virtual_gates_values[virt_channel] != 0:
						real_channel.add_reference_channel(virtual.virtual_gate_names[j], 
							getattr(self, virtual_gates.virtual_gate_names[j]),
							virtual_gates_values[j])


		# define virtual IQ channels
		for IQ_channels_obj in IQ_channels_objs:
			for virtual_channel_name in IQ_channels_obj.virtual_channel_map:
				setattr(self, virtual_channel_name.channel_name, segment_IQ(virtual_channel_name.channel_name, virtual_channel_name.reference_frequency))
				self.channels.append(virtual_channel_name)

			# set up maping to real IQ channels:
			for IQ_real_channel_info in IQ_channels_obj.IQ_channel_map:
				real_channel = getattr(self, IQ_real_channel_info.channel_name)
				for virtual_channel_name in IQ_channels_obj.virtual_channel_map:
					virtual_channel = getattr(self, virtual_channel_name.channel_name)
					real_channel.add_IQ_channel(IQ_channels_obj.LO, virtual_channel_name.channel_name, virtual_channel, IQ_real_channel_info.IQ_comp, IQ_real_channel_info.image)

			# set up markers
			for marker_info in IQ_channels_obj.markers:
				real_channel_marker = getattr(self, marker_info.Marker_channel)
				
				for virtual_channel_name in IQ_channels_obj.virtual_channel_map:
					virtual_channel = getattr(self, virtual_channel_name.channel_name)
					real_channel_marker.add_reference_marker_IQ(virtual_channel, marker_info.pre_delay, marker_info.post_delay)

	@property
	def shape(self):
		'''
		get combined shape of all the waveforms
		'''
		my_shape = (1,)
		for i in self.channels:
			dim = getattr(self, i).shape
			my_shape = find_common_dimension(my_shape, dim)

		return my_shape
		
	@property
	def last_mod(self):
		time = datetime.datetime.utcfromtimestamp(0)
		for i in self.channels:
			if getattr(self, i, segment_single()).last_edit > time:
				time = getattr(self, i, segment_single()).last_edit
		return time

	@property
	def total_time(self,):
		'''
		get the total time that will be uploaded for this segment to the AWG
		Returns:
			times (np.ndarray) : numpy array with the total time (maximum of all the channels), for all the different loops executed.
		'''
		self.extend_dim()

		shape = list(getattr(self, self.channels[0]).data.shape)
		n_channels = len(self.channels)
		
		time_data = np.empty([n_channels] + shape)
		
		for i in range(len(self.channels)):
			time_data[i] = getattr(self, self.channels[i]).total_time

		times = np.amax(time_data, axis = 0)

		return times

	@property
	def Vmin_max_data(self):
		if self.prev_upload < self.last_mod:

			for i in range(len(self.channels)):
				self._Vmin_max_data[self.channels[i]]['v_min'] = getattr(self,self.channels[i]).v_min
				self._Vmin_max_data[self.channels[i]]['v_max'] = getattr(self,self.channels[i]).v_max

		return self._Vmin_max_data

	def append(self, other, time=None):
		'''
		append other segments the the current ones in the container.
		Args:
			other (segment_container) : other segment to append
		'''
		if not isinstance(other, segment_container):
			raise TypeError("segment_container object expected. Did you supply a single segment?")
		if time == None:
			times = self.total_time

			time = lp.loop_obj()
			time.add_data(times, list(range(len(times.shape)-1, -1,-1)))

		for i in self.channels:
			segment = getattr(self, i)
			segment.append(getattr(other, i), time)

	def slice_time(self, start, stop):
		"""
		slice time in a segment container
		Args:
			start (double) : start time of the slice
			stop (double) : stop time of the slice

		The slice_time function allows you to cut all the waveforms in the segment container in different sizes.
		This function should be handy for debugging, example usage would be, 
		You are runnning an algorithm and want to check what the measurement outcomes are though the whole algorithm.
		Pratically, you want to know
			0 -> 10ns (@10 ns still everything as expected?)
			0 -> 20ns
			0 -> ...
		This function would allow you to do that, e.g. by calling segment_container.cut_segment(0, lp.linspace(10,100,9))
		"""
		for i in self.channels:
			segment = getattr(self, i)
			segment.slice_time(start, stop)

	def reset_time(self, extend_only = False):
		'''
		Args:
			extend_only (bool) : will just extend the time in the segment and not reset it if set to true [do not use when composing wavoforms...].
			
		Allings all segments togeter and sets the input time to 0,
		e.g. , 
		chan1 : waveform until 70 ns
		chan2 : waveform until 140ns
		-> totaltime will be 140 ns,
		when you now as a new pulse (e.g. at time 0, it will actually occur at 140 ns in both blocks)
		'''
		
		times = self.total_time

		loop_obj = lp.loop_obj()
		loop_obj.add_data(times, list(range(len(times.shape)-1, -1,-1)))

		for i in self.channels:
			segment = getattr(self, i)
			segment.reset_time(loop_obj, extend_only)

	def get_waveform(self, channel, index = [0], pre_delay=0, post_delay = 0, sample_rate=1e9):
		'''
		function to get the raw data of a waveform,
		inputs:
			channel (str) : channel name of the waveform you want
			index (tuple) :
			pre_delay (int) : extra offset in from of the waveform (start at negative time) (for a certain channel, as defined in channel delays)
			post_delay (int) : time gets appended to the waveform (for a certain channel)
		returns:
			np.ndarray[ndim=1, dtype=double] : waveform as a numpy array
		'''
		return getattr(self, channel).get_segment(index, pre_delay, post_delay, sample_rate)

	def extend_dim(self, shape=None, ref = False):
		'''
		extend the dimensions of the waveform to a given shape.
		Args:
			shape (tuple) : shape of the new waveform
			ref (bool) : put to True if you want to extend the dimension by using pointers instead of making full copies.
		If referencing is True, a pre-render will already be performed to make sure nothing is rendered double. 
		'''
		if shape is None:
			shape = self.shape

		for i in self.channels:
			if self.render_mode == False:
				getattr(self, i).data = update_dimension(getattr(self, i).data, shape, ref)
			
			if getattr(self, i).type == 'render' and self.render_mode == True:
				getattr(self, i)._pulse_data_all = update_dimension(getattr(self, i)._pulse_data_all, shape, ref)

	def enter_rendering_mode(self):
		'''
		put the segments into rendering mode, which means that they cannot be changed. All segments will get their final length at this moment.
		'''
		self.reset_time()
		self.render_mode = True
		for i in self.channels:
			getattr(self, i).render_mode =  True
			# make a pre-render all all the pulse data (e.g. compose channels, do not render in full).
			if getattr(self, i).type == 'render':
				getattr(self, i).pulse_data_all

	def exit_rendering_mode(self):
		'''
		exit rendering mode and clear all the ram that was used for the rendering.
		'''
		self.render_mode = False
		for i in self.channels:
			getattr(self, i).render_mode =  False
			getattr(self, i)._pulse_data_all = None

if __name__ == '__main__':
	import pulse_lib.segments.utility.looping as lp
	a = segment_container(["a", "b"])
	b = segment_container(["a", "b"])

	b.a.add_block(0,lp.linspace(50,100,10),100)
	b.a.reset_time()

	b.a.add_block(20,lp.linspace(50,100,10),100)

	a.append(b,10)

	a.slice_time(0,lp.linspace(80,100,10))
	print(a.shape)
	print(a.a.data[2,2,2].my_pulse_data)