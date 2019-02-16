import pulse_lib.segments.segments_base as seg_base
import pulse_lib.segments.segments_IQ as seg_IQ
import pulse_lib.segments.looping as lp

from pulse_lib.segments.data_handling_functions import find_common_dimension, update_dimension
import uuid

import numpy as np
import datetime

"""
TODO : 
implement reset time for all channels.
force right dimensions.

"""

class segment_container():
	'''
    Class containing all the single segments for for a series of channels.
	This is a organisational class.
	Class is capable of checking wheather upload is needed.
	Class is capable of termining what volatages are required for each channel.
	Class returns vmin/vmax data to awg object
	Class returns upload data as a int16 array to awg object.
    '''
	def __init__(self, real_channels, virtual_gates = None, IQ_channels=None):
		# physical channels
		self.r_channels = []
		# physical + virtual channels
		self.channels = []
		self.virtual_gates = virtual_gates
		self.render_mode = False
		self.id = uuid.uuid4()
		self._Vmin_max_data = dict()

		for i in real_channels:
			self._Vmin_max_data[i] = {"v_min" : None, "v_max" : None}
		
		self.prev_upload = datetime.datetime.utcfromtimestamp(0)

		
		# Not superclean should be in a different namespace.
		for i in real_channels:
			setattr(self, i, seg_base.segment_single(i))
			self.channels.append(i)
			self.r_channels.append(i)

		if virtual_gates is not None:
			# make segments for virtual gates.
			for i in self.virtual_gates['virtual_gates_names_virt']:
				setattr(self, i, seg_base.segment_single(i, 'virtual'))
				self.channels.append(i)

			# add reference in real gates.
			for i in range(len(self.virtual_gates['virtual_gates_names_virt'])):
				current_channel = getattr(self, self.virtual_gates['virtual_gates_names_real'][i])
				virtual_gates_values = self.virtual_gates['virtual_gate_matrix'][i,:]

				for virt_channel in range(len(self.virtual_gates['virtual_gates_names_virt'])):
					if virtual_gates_values[virt_channel] != 0:
						current_channel.add_reference_channel(self.virtual_gates['virtual_gates_names_virt'][virt_channel], 
							getattr(self, self.virtual_gates['virtual_gates_names_virt'][virt_channel]),
							virtual_gates_values[virt_channel])

		if IQ_channels is not None:
			for i in range(len(IQ_channels['vIQ_channels'])):
				setattr(self, IQ_channels['vIQ_channels'][i], seg_IQ.segment_single_IQ(IQ_channels['vIQ_channels'][i], IQ_channels['LO_freq'][i]))
				self.channels.append(IQ_channels['vIQ_channels'][i])

			for i in range(len(IQ_channels['rIQ_channels'])):
				I_channel = getattr(self, IQ_channels['rIQ_channels'][i][0])
				Q_channel = getattr(self, IQ_channels['rIQ_channels'][i][1])
				I_channel.add_IQ_ref_channel(IQ_channels['vIQ_channels'][i],
					getattr(self, IQ_channels['vIQ_channels'][i]), 'I')
				Q_channel.add_IQ_ref_channel(IQ_channels['vIQ_channels'][i],
					getattr(self, IQ_channels['vIQ_channels'][i]), 'Q')


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
	def last_mod(self):
		time = datetime.datetime.utcfromtimestamp(0)
		for i in self.channels:
			if getattr(self, i, segment_single()).last_edit > time:
				time = getattr(self, i, segment_single()).last_edit
		return time

	@property
	def Vmin_max_data(self):
		if self.prev_upload < self.last_mod:

			for i in range(len(self.channels)):
				self._Vmin_max_data[self.channels[i]]['v_min'] = getattr(self,self.channels[i]).v_min
				self._Vmin_max_data[self.channels[i]]['v_max'] = getattr(self,self.channels[i]).v_max

		return self._Vmin_max_data

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
		
	def clear_chache(self):
		raise NotImplemented

	@property
	def shape(self):
		'''
		get combined shape of all the waveforms
		'''
		my_shape = (1,)
		for i in self.channels:
			dim = getattr(self, i).data.shape
			my_shape = find_common_dimension(my_shape, dim)

		return my_shape

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

	def append(self, other):
		'''
		append other segments the the current one.
		Args:
			other (segment_container) : other segment to append
		'''
		pass