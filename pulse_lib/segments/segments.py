import pulse_lib.segments.segments_base as seg_base
import pulse_lib.segments.segments_IQ as seg_IQ
from pulse_lib.segments.data_handling_functions import find_common_dimension, update_dimension

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
	def __init__(self, name, real_channels, virtual_gates = None, IQ_channels=None):
		# physical channels
		self.r_channels = []
		# physical + virtual channels
		self.channels = []
		self.virtual_gates = virtual_gates
		self.name = name
		self._Vmin_max_data = dict()

		for i in real_channels:
			self._Vmin_max_data[i] = {"v_min" : None, "v_max" : None}
		
		self.prev_upload = datetime.datetime.utcfromtimestamp(0)

		
		# Not superclean should be in a different namespace.
		for i in real_channels:
			setattr(self, i, seg_base.segment_single())
			self.channels.append(i)
			self.r_channels.append(i)

		if virtual_gates is not None:
			# make segments for virtual gates.
			for i in self.virtual_gates['virtual_gates_names_virt']:
				setattr(self, i, seg_base.segment_single())
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
				setattr(self, IQ_channels['vIQ_channels'][i], seg_IQ.segment_single_IQ(IQ_channels['LO_freq'][i]))
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
		self.__extend_dim_all_waveforms()

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

	def reset_time(self):
		'''
		Allings all segments togeter and sets the input time to 0,
		e.g. , 
		chan1 : waveform until 70 ns
		chan2 : waveform until 140ns
		-> totaltime will be 140 ns,
		when you now as a new pulse (e.g. at time 0, it will actually occur at 140 ns in both blocks)
		'''
		raise NotImplemented

	def get_waveform(self, channel, Vpp_data, sequence_time, pre_delay=0, post_delay = 0, return_type = np.double):
		'''
		function to get the raw data of a waveform,
		inputs:
			channel: channel name of the waveform you want
			Vpp_data: contains peak to peak voltage and offset for each channel
			sequence time: efffective time in the sequence when the segment starts, this can be important for when using mulitple segments with IQ modulation.
			pre_delay: extra offset in from of the waveform (start at negative time) (for a certain channel, as defined in channel delays)
			post_delay: time gets appended to the waveform (for a certain channel)
			return type: type of the wavefrom (must be numpy compatible). Here number between -1 and 1.
		returns:
			waveform as a numpy array with the specified data type.
		'''
		# self.prep4upload()


		# upload_data = np.empty([int(self.total_time) + pre_delay + post_delay], dtype = return_type)
		
		waveform_raw = getattr(self, channel).get_segment(self.total_time, sequence_time, pre_delay, post_delay)

		# chan_number = None
		# for i in range(len(self.channels)):
		# 	if self.channels[i] == channel:
		# 		chan_number = i

		# do not devide by 0 (means channels is not used..)
		if Vpp_data[channel]['v_pp'] == 0:
			Vpp_data[channel]['v_pp'] = 1
			
		# normalise according to the channel, put as 
		upload_data = ((waveform_raw - Vpp_data[channel]['v_off'])/Vpp_data[channel]['v_pp']).astype(return_type)

		return upload_data

	def clear_chache(self):
		raise NotImplemented

	def __extend_dim_all_waveforms(self):
		"""
		function to make sure that all the waveforms have the same dimentionality.
		Note that the mode here is copy and not referencing.
		"""

		# find global dimension
		my_dimension = (1,)
		for i in self.channels:
			dim = getattr(self, i).data.shape
			my_dimension = find_common_dimension(my_dimension, dim)

		# now update the size
		for i in self.channels:
			getattr(self, i).data = update_dimension(getattr(self, i).data, my_dimension)

