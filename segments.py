import numpy as np
import datetime
import matplotlib.pyplot as plt
import copy
class segment_container():
	'''
    Class containing all the single segments for for a series of channels.
	This is a organisational class.
	Class is capable of checking wheather upload is needed.
	Class is capable of termining what volatages are required for each channel.
	Class returns vmin/vmax data to awg object
	Class returns upload data as a int16 array to awg object.
    '''
	def __init__(self, name, channels, virtual_gates = None):
		self.channels = channels
		self.virtual_gates = virtual_gates
		self.name = name
		self._Vmin_max_data = dict()

		for i in self.channels:
			self._Vmin_max_data[i] = {"v_min" : None, "v_max" : None}
		
		self.prev_upload = datetime.datetime.utcfromtimestamp(0)

		# self.vpp_data = dict()
		# for i in self.channels:
		# 	self.vpp_data[i] = {"V_min" : None, "V_max" : None}
		
		# Not superclean should be in a different namespace.
		for i in self.channels:
			setattr(self, i, segment_single())

		if virtual_gates is not None:
			# make segments for virtual gates.
			for i in self.virtual_gates['virtual_gates_names_virt']:
				setattr(self, i, segment_single())

			# add reference in real gates.
			for i in range(len(self.virtual_gates['virtual_gates_names_virt'])):
				current_channel = getattr(self, self.virtual_gates['virtual_gates_names_real'][i])
				virtual_gates_values = self.virtual_gates['virtual_gate_matrix'][i,:]

				for virt_channel in range(len(self.virtual_gates['virtual_gates_names_virt'])):
					if virtual_gates_values[virt_channel] != 0:
						current_channel.add_reference_channel(self.virtual_gates['virtual_gates_names_virt'][virt_channel], 
							getattr(self, self.virtual_gates['virtual_gates_names_virt'][virt_channel]),
							virtual_gates_values[virt_channel])
	@property
	def total_time(self):
		time_segment = 0
		for i in self.channels:
			if time_segment <= getattr(self, i).total_time:
				time_segment = getattr(self, i).total_time

		return time_segment

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
		maxtime = 0
		for i in self.channels:
			k = getattr(self, i)
			t = k.get_total_time()
			if t > maxtime:
				maxtime = t
		for i in self.channels:
			getattr(self, i).starttime = maxtime

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

	def clear_chache():
		return

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
		self._last_edit = datetime.datetime.now()
		self.my_pulse_data = np.zeros([1,2])
		self.sin_data = []
		self.numpy_data = []
		self.last = None
		self.IQ_data = [] #todo later.
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

		self.my_pulse_data = self._add_up_pulse_data(arr)

	@last_edited
	def add_block(self,start,stop, amplitude):
		'''
		add a block pulse on top of the existing pulse.
		'''
		if start != 0:
			pulse = np.array([[0,0], [start + self.starttime, 0], [start + self.starttime,amplitude], [stop + self.starttime, amplitude], [stop + self.starttime, 0]])
		else:
			pulse = np.array([[start + self.starttime, 0], [start + self.starttime,amplitude], [stop + self.starttime, amplitude], [stop + self.starttime, 0]])
		self.my_pulse_data = self._add_up_pulse_data(pulse)

	@last_edited
	def wait(self, wait):
		'''
		wait for x ns after the lastest wave element that was programmed.
		'''
		amp_0 = self.my_pulse_data[-1,1]
		t0 = self.total_time
		pulse = [[wait+t0, amp_0]]
		self.add_pulse(pulse)

	@last_edited
	def add_sin(self, start, stop, amp, freq, phase_offset=0):
		'''
		add a sinus to the current segment, parameters should be self exlenatory.
		The pulse will have a relative phase (as this is needed to all IQ work).
		'''
		self.sin_data.append(
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
		my_pulse_data_copy = copy.copy(self.my_pulse_data)

		if my_pulse_data_copy[-1,0] < self.total_time:
			my_pulse_data_copy = np.append(my_pulse_data_copy, [[self.total_time, my_pulse_data_copy[-1,1]]], axis=0)

		front_pulse_corr = None
		back_pulse_corr = None
		# check if there is twice the same starting number 
		
		if self.my_pulse_data[0,0] == self.my_pulse_data[1,0]:
			front_pulse_corr = my_pulse_data_copy[0]
			my_pulse_data_copy = my_pulse_data_copy[1:]

		if self.my_pulse_data[-1,0] == self.my_pulse_data[-2,0]:
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

			for sin_data_item in self.sin_data:
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

		self.my_pulse_data = pulse_data
		self.sin_data = sin_data

	@last_edited
	def add_np(self,start, array):
		raise NotImplemented

	def __add__(self, other):
		'''
		define addition operator for segment_single
		'''
		new_segment = segment_single()
		if type(other) is segment_single:
			new_segment.my_pulse_data = self._add_up_pulse_data(other.my_pulse_data)
			sin_data = self.sin_data
			sin_data.extend(other.sin_data)
			new_segment.sin_data = sin_data
		elif type(other) == int or type(other) == float:
			new_pulse = self.my_pulse_data
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
			new_pulse = self.my_pulse_data
			new_pulse[:,1] *= other
			new_segment.my_pulse_data = new_pulse
			sin_data = []
			for i in self.sin_data:
				my_sin = i
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
		new_pulse --> defailt format as in the add_pulse function
		'''
		my_pulse_data_copy = self.my_pulse_data
		# step 1: make sure both pulses have the same length
		if self.total_time < new_pulse[-1,0]:
			to_insert = [[new_pulse[-1,0],my_pulse_data_copy[-1,1]]]
			my_pulse_data_copy = self._insert_arrays(my_pulse_data_copy, to_insert, len(my_pulse_data_copy)-1)
		elif self.total_time > new_pulse[-1,0]:
			to_insert = [[my_pulse_data_copy[-1,0],new_pulse[-1,1]]]
			new_pulse = self._insert_arrays(new_pulse, to_insert, len(new_pulse)-1)


		# step 2: construct time indices of new array
		times = np.unique(list(self.my_pulse_data[:,0]) + list(new_pulse[:,0]))

		new_times = []
		for time in times:
			n_times = max(len(np.where(time == my_pulse_data_copy[:,0])[0]), len(np.where(time == new_pulse[:,0])[0]))
			new_times.extend([time]*n_times)
		new_times = np.array(new_times)
		
		my_pulse_data_tmp = np.zeros([len(new_times),2])
		my_pulse_data_tmp[:,0] = new_times
		new_pulse_tmp = np.zeros([len(new_times),2])
		new_pulse_tmp[:,0] = new_times

		# step 3: add missing values to tmp vars.
		my_pulse_data_tmp = interpolate_pulse(my_pulse_data_tmp, my_pulse_data_copy)
		new_pulse_tmp = interpolate_pulse(new_pulse_tmp, new_pulse)

		# step 4 contruct final pulse
		final_pulse = np.zeros([len(new_times),2])
		final_pulse[:,0] = new_times
		final_pulse[:,1] +=  my_pulse_data_tmp[:,1]  + new_pulse_tmp[:,1]

		return final_pulse

	@property
	def total_time(self,):
		total_time = 0
		for sin_data_item in self.sin_data:
			if sin_data_item['stop_time'] > total_time:
				total_time = sin_data_item['stop_time']

		if self.my_pulse_data[-1,0] > total_time:
			total_time = self.my_pulse_data[-1,0]
		return total_time

	def get_total_time(self, segment):
		return self.my_pulse_data[-1,0]

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
		t, wvf = self._generate_segment(points, t_start, pre_delay, post_delay)
		return wvf

	@property
	def v_max(self):
		return np.max(self.my_pulse_data[:,1])

	@property
	def v_min(self):
		return np.min(self.my_pulse_data[:,1])

	@property
	def pulse_data_all(self):
		if self.last_edit > self.last_render:
			self._pulse_data_all = copy.copy(self)

			for ref_chan in self.reference_channels:
				self._pulse_data_all += ref_chan['segment']*ref_chan['multiplication_factor']

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
		plt.show()

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
		
class marker_single():
	def __init__(self):
		self.type = 'default'
		self.swing = False
		self.latest_time = 0

		self.my_pulse_data = np.zeros([1,2])

	def add(self, start, stop):
		self.my_pulse_data = np.append(self.my_pulse_data, [[start,0],[start,1],[stop,1],[stop,0]])

def interpolate_pulse(new_pulse, old_pulse):
	'''
	interpolate value in new_pulse that are not present in the old pulse array
	Args: 
		new_pulse: 2D array containin the timings and amplitudes desciribing the pulse
		old_pulse: 2D array, but from the old one
	Return:
		new_pulse, with interpolated values

	# note this only works if old pulse and new pulse do have the same starting argument.
	'''
	skip_next_iteration = False

	for i in range(len(new_pulse)):
		if skip_next_iteration == True:
			skip_next_iteration = False
			continue

		times = np.where(new_pulse[i,0] == old_pulse[:,0])[0]
		if len(times) == 1:
			new_pulse[i,1] = old_pulse[times[0],1]
			if i != len(new_pulse)-1:
				if new_pulse[i+1,0] == new_pulse[i,0]:
					new_pulse[i+1,1] = old_pulse[times[0],1]
					skip_next_iteration = True
		elif len(times) == 2:
			new_pulse[i:i+2,:] = old_pulse[times,:]
			skip_next_iteration = True
		else:
			prev_point = np.where(old_pulse[:,0] <= new_pulse[i,0])[0][-1]
			next_point = np.where(old_pulse[:,0] >= new_pulse[i,0])[0][ 0]

			interpolation = calc_value_point_in_between(old_pulse[prev_point], old_pulse[next_point], new_pulse[i,0])
			new_pulse[i,1] = interpolation

	return new_pulse

def calc_value_point_in_between(ref1, ref2, time):
		'''
		calculate amplitude of waveform at the certain time 
		Args:
			ref1 : time1 and amplitude1  (part of a pulse object)
			ref2 : time2 and amplitude2 at a different time (part of a pulse object)
			time : time at which you want to know to ampltitude (should be inbetween t1 and t2)
		Returns:
			amplitude at time 
		'''
		a = (ref2[1]- ref1[1])/(ref2[0]- ref1[0])
		c = ref1[1] - a*ref1[0]

		return a*time + c


if __name__ == '__main__':
	# s = segment_single()
	
	# amp = np.linspace(-150,150,50)

	# step_time = 100
	# t0= 0
	# for i in range(50):
	# 	s.add_block(t0, t0 + step_time, amp[i])
	# 	t0 += step_time
	# s.repeat(50)
	# # s2 = s +s
	# s._generate_segment()
	# # print(s3)
	# # s2.plot_segment()
	# # plt.show()

	awg_channels = ['P1','P2','P3','P4','P5','B0','B1','B2','B3','B4','B5','G1','I_MW', 'Q_MW', 'M1', 'M2']
	awg_virtual_channels = {'virtual_gates_names_virt' : ['vP1','vP2','vP3','vP4','vP5','vB0','vB1','vB2','vB3','vB4','vB5'],
									 'virtual_gates_names_real' : ['P1','P2','P3','P4','P5','B0','B1','B2','B3','B4','B5'],
									 'virtual_gate_matrix' : np.eye(11)}
	awg_virtual_channels['virtual_gate_matrix'][0,1] = 0.1
	awg_virtual_channels['virtual_gate_matrix'][0,2] = 0.1
	a = segment_container('test', awg_channels, awg_virtual_channels)

	print(a.P1.reference_channels)
	print(a.P1.last_edit)
	print(a.P1.pulse_data_all)
	# a.vP1.add_block(1,10,50)
	# a.vP2.add_block(20,30,50)
	a.vP3.add_block(10,60,50)
	a.vP3.add_sin(20,500, 1, 1e7)
	a.vP3.repeat(10)
	print(a.P1.last_edit)

	# print(a.P1.reference_channels[1]['name'])
	# print(a.P1.reference_channels[1]['segment'].my_pulse_data)

	a.P1.plot_segment()