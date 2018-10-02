import numpy as np
import datetime
import matplotlib.pyplot as plt

class segment_container():
	'''
    Class containing all the single segments for for a series of channels.
	This is a organisational class.
	Class is capable of checking wheather upload is needed.
	Class is capable of termining what volatages are required for each channel.
	Class returns vmin/vmax data to awg object
	Class returns upload data as a int16 array to awg object.
    '''
	def __init__(self, name, channels):
		self.channels = channels
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


	# def prep4upload(self):
	# 	# make waveform (in chache) (only if needed)
	# 	t_tot = self.total_time

	# 	if self.prev_upload < self.last_mod or self.waveform_cache is None:
	# 		self.waveform_cache = np.empty([len(self.channels), int(t_tot)])

	# 		for i in range(len(self.channels)):
	# 			self.waveform_cache[i,:] = getattr(self, self.channels[i]).get_segment(t_tot)


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
		args[0].last_edit = datetime.datetime.now()
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
		self.last_edit = datetime.datetime.now()
		self.my_pulse_data = np.zeros([1,2])
		self.last = None
		self.IQ_data = [] #todo later.
		self.unique = False

	@last_edited
	def add_pulse(self,array):
		'''
		format :: 
		[[t0, Amp0],[t1, Amp1]]
		'''
		arr = np.asarray(array)
		arr[:,0] = arr[:,0] + self.starttime 

		self.my_pulse_data = np.append(self.my_pulse_data,arr, axis=0)

	@last_edited
	def add_block(self,start,stop, amplitude):
		amp_0 = self.my_pulse_data[-1,1]
		pulse = [ [start, amp_0], [start,amplitude], [stop, amplitude], [stop, amp_0]]
		self.add_pulse(pulse)

	@last_edited
	def wait(self, wait):
		amp_0 = self.my_pulse_data[-1,1]
		t0 = self.my_pulse_data[-1,0]
		pulse = [[wait+t0, amp_0]]
		self.add_pulse(pulse)

	def repeat(self, number):
		return

	def add_np(self,start, array):
		return

	def add_IQ_pair():
		return

	def add():
		return
	
	@property
	def total_time(self,):
		return self.my_pulse_data[-1,0]

	def get_total_time(self):
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


	def _generate_segment(self, t_tot= None, t_start = 0, pre_delay = 0, post_delay = 0):
		# TODO implement t_start feature.

		# 1 make base sequence:
		if t_tot is None:
			t_tot = self.total_time

		times = np.linspace(-pre_delay, int(t_tot-1 + post_delay), int(t_tot + pre_delay + post_delay))
		my_sequence = np.zeros([int(t_tot + pre_delay + post_delay)])


		for i in range(0,len(self.my_pulse_data)-1):
			my_loc = np.where(times < self.my_pulse_data[i+1,0])[0]
			my_loc = np.where(times[my_loc] >= self.my_pulse_data[i,0])[0]

			if my_loc.size == 0:
				continue;

			end_voltage = self.my_pulse_data[i,1] + \
				(self.my_pulse_data[i+1,1] - self.my_pulse_data[i,1])* \
				(times[my_loc[-1]] - self.my_pulse_data[i,0])/ \
				(self.my_pulse_data[i+1,0]- self.my_pulse_data[i,0])

			start_stop_values = np.linspace(self.my_pulse_data[i,1],end_voltage,my_loc.size);
			my_sequence[my_loc] = start_stop_values;
		#max_init = np.max(np.abs(my_sequence))
		#my_sequence = flt.gaussTaperFilter(my_sequence)
		#my_sequence /= np.max(np.abs(my_sequence))
		return times, my_sequence

	def plot_segment(self):
		x,y = self._generate_segment()
		plt.plot(x,y)
		plt.show()
		
class marker_single():
	def __init__(self):
		self.type = 'default'
		self.swing = False
		self.latest_time = 0

		self.my_pulse_data = np.zeros([1,2])

	def add(self, start, stop):
		self.my_pulse_data = np.append(self.my_pulse_data, [[start,0],[start,1],[stop,1],[stop,0]])
