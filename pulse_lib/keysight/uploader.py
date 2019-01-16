import threading as th
import numpy as np
import time
from pulse_lib.keysight.AWG_memory_manager import Memory_manager
from pulse_lib.keysight.uploader_core.uploader import waveform_upload_chache
def mk_thread(function):
    def wrapper(*args, **kwargs):
        thread = th.Thread(target=function, args=args, kwargs=kwargs)
        thread.start()
        return thread
    return wrapper



class keysight_uploader():
	"""
	Object responsible for uploading waveforms to the keysight AWG in a timely fashion.
	"""
	def __init__(self, AWGs, channel_locations, channel_delays, channel_compenstation_limits):
		'''
		Initialize the keysight uploader. 
		Args:
			AWGs (list<QcodesIntrument>) : list with AWG's
			channel_locations (dict): dict with channel and AWG+channel location
			channel_compenstation_limits (dict) : dict with channel name as key and tuple as value with lower and upper limit
		Returns:
			None
		'''
		self.memory_allocation = dict()
		self.upload = True
		# TODO reinit memory on start-up
		# for i in AWGs:
		# 	self.memory_allocation[i.name] = Memory_manager()
		self.channel_map = channel_locations
		self.channel_delays = channel_delays
		self.channel_compenstation_limits = channel_compenstation_limits
		self.upload_queue = []
		self.upload_done = []

	def add_upload_job(self, job):
		'''
		add a job to the uploader.
		Args:
			job (upload_job) : upload_job object that defines what needs to be uploaded and possible post processing of the waveforms (if needed)
		'''
		self.upload_queue.append(job)


	def get_segment_data(self, job):
		'''
		get the segment numbers where an upload has been performed.
		Args:
			job (type?) :
		Returns:
			dict <array <int> >: map with with the channels where the segments have been uploaded
		'''
		pass

	def __get_new_job(self):
		'''
		get the next job that needs to be uploaded.

		'''
		# get item in the queue with the highest prioriry
		priority = -1
		for i in self.upload_queue:
			if i.priority > priority:
				priority = i.priority 

		# get the job
		for i in range(len(self.upload_queue)):
			if self.upload_queue[i].priority == priority:
				job = self.upload_queue[i]
				self.upload_queue.pop(i)
				return job

		return None

	def __segment_AWG_memory(self):
		'''
		Generates segments in the memory in the Keysight AWG.
		'''
	@mk_thread
	def uploader(self):
		'''
		Class taking care of putting the waveform on the right AWG. This is a continuous thread that is run in the background.

		Steps:
		1) get all the upload data
		2) perform DC correction (if needed)
		3) perform DSP correction (if needed)
		4) convert data in an aprropriate upload format (c++)
		4) upload all data (c++)
		5) write in the job object the resulting locations of sequences that have been uploaded.

		'''

		while self.upload == True:
			job = self.__get_new_job()

			if job is None:
				# wait one ms and continue
				time.sleep(0.001)
				self.upload = False
				continue
			
			start = time.time()

			# 1) get all the upload data -- construct object to hall the rendered data
			waveform_cache = dict()
			for i in self.channel_map:
				waveform_cache[i] = waveform_upload_chache(self.channel_compenstation_limits[i])

			

			pre_delay = 0
			post_delay = 0
			print("\n\n\n new call on pulse data\n\n")
			for i in range(len(job.sequence)):

				seg = job.sequence[i][0]
				n_rep = job.sequence[i][1]
				prescaler = job.sequence[i][2]
				
				# TODO add precaler in as sample rate
				for channel in self.channel_map:
					if i == 0:
						pre_delay = self.channel_delays[channel][0]
					if i == len(job.sequence) -1:
						post_delay = self.channel_delays[channel][1]
					
					sample_rate = 1e9
					wvf = seg.get_waveform(channel, job.index, pre_delay, post_delay, sample_rate)

					integral = 0
					if job.neutralize == True:
						integral = getattr(seg, channel).integrate(job.index, pre_delay, post_delay, sample_rate)

					vmin = getattr(seg, channel).v_min(job.index, sample_rate)
					vmax = getattr(seg, channel).v_max(job.index, sample_rate)
					
					waveform_cache[channel].add_data(wvf, (vmin, vmax), integral)

					pre_delay = 0
					post_delay = 0

			
			end1 = time.time()
			# 2) perform DC correction (if needed)
			'''
			Steps: [TODO : best way to include sample rate here? (by default now 1GS/s)]
				a) calculate total compensation time needed (based on given boundaries).
				b) make sure time is modulo 10 (do that here?)
				c) add segments with the compenstated pulse for the given total time.
			'''
			compensation_time = 0
			wvf_npt = 0
			for chan in waveform_cache:
				if waveform_cache[chan].compensation_time > compensation_time:
					compensation_time = waveform_cache[chan].compensation_time 
					wvf_npt = waveform_cache[chan].npt
			# make sure we have modulo 10 time
			total_pt = compensation_time + wvf_npt
			mod = total_pt%10
			if mod != 0:
				total_pt += 10-mod
			compensation_time = total_pt - wvf_npt

			#generate the compensation
			for chan in waveform_cache:
				waveform_cache[chan].generate_voltage_compensation(compensation_time)
			end = time.time()
			print("time needed to render and compenstate",end - start)
			print("rendering = ", end1 - start)
			print("compensation = ", end - end1)
			
			# 3) DSP correction
			# TODO later

			# 4) 

class upload_job(object):
	"""docstring for upload_job"""
	def __init__(self, sequence, index, seq_id, neutralize=True, priority=0):
		'''
		Args:
			sequence (list of list): list with list of the sequence, number of repetitions and prescalor (// upload rate , see keysight manual)
			index (tuple) : index that needs to be uploaded
			seq_id (uuid) : if of the sequence
			neutralize (bool) : place a neutralizing segment at the end of the upload
			priority (int) : priority of the job (the higher one will be excuted first)
		'''
		# TODO make custom function for this. This should just extend time, not reset it.
		self.sequence = sequence
		self.id = seq_id
		self.index = index
		self.neutralize = True
		self.priority = priority
		self.DSP = False
	
	def add_dsp_function(self, DSP):
		self.DSP =True
		self.DSP_func = DSP



class waveform_upload_chache():
	"""object that holds some cache for uploads and does some basic calculations"""
	def __init__(self, compenstation_limit):
		self.data = []
		self.min_max_voltage = (0,0)
		self.compenstation_limit = compenstation_limit

	def add_data(self, wvf, integral, v_min_max):
		'''
		wvf (np.ndarray[ndim = 1, dtype=double]) : waveform
		integral (double) : integrated value of the waveform
		v_min_max (tuple) : maximum/minimum voltage of the current segment
		'''
		data = {'wvf' : wvf, 'integral' : integral, 'v_min_max' : v_min_max}
		self.data.append(data)

	@property
	def integral(self):
		integral = 0
		for i in self.data:
			integral += i['integral']
		return integral

	@property
	def compensation_time(self):
		'''
		return the minimal compensation time that is needed.
		Returns:
			compensation_time : minimal duration that is needed for the voltage compensation
		'''
		comp_time = self.integral
		if comp_time <= 0:
			return -comp_time / self.compenstation_limit[1]
		else:
			return -comp_time / self.compenstation_limit[0]

	def get_min_max(self):
		'''
		get min/maximum voltage that is saved in this object.
		'''
		pass

	def generate_voltage_compensation(self, time):
		'''
		make a voltage compenstation pulse of time t
		Args:
			time (double) : time of the compenstation in ns
		'''
		if round(time) == 0:
			voltage = 0
		else:
			voltage = self.integral/round(time)

		wvf = np.full((int(round(time)),), voltage)
		data = {'wvf' : wvf, 'integral' : -self.integral, 'v_min_max' : (voltage, voltage)}
		self.data.append(data)


		
		


# class sequencer():
# 	def __init__(self, awg_system, channel_delays, segment_bin):
# 		self.awg = awg_system
# 		self.segment_bin = segment_bin
# 		self.channels = segment_bin.channels
# 		self.channel_delays = channel_delays
# 		self.sequences = dict()

# 	def add_sequence(self, name, sequence):
# 		self.sequences[name] = sequence

# 	def start_sequence(self, name):
# 		self.get_sequence_upload_data(name)
# 		self.awg.upload(self.sequences[name], self.get_sequence_upload_data(name))
# 		self.awg.start()

# 	def get_sequence_upload_data(self, name):
# 		'''
# 		Function that generates sequence data per channel.
# 		It will also assign unique id's to unique sequences (sequence that depends on the time of playback). -> mainly important for iq mod purposes.
# 		structure:
# 			dict with key of channel:
# 			for each channels list of sequence:
# 				name of the segments,
# 				number of times to play
# 				uniqueness -> segment is reusable?
# 				identifiers for marking differnt locations in the ram of the awg.
		
# 		'''
# 		upload_data = dict()
# 		# put in a getter to make sure there is no error -- it exists...
# 		seq = self.sequences[name]

# 		for chan in self.channels:
# 			sequence_data_single_channel = []
# 			num_elements = len(seq)

# 			for k in range(len(seq)):
# 				segment_play_info = seq[k]

# 				# if starting segment or finishing segment, here there should be added the delay info.
# 				pre_delay, post_delay = (0,0)

# 				if k == 0:
# 					pre_delay = self.get_pre_delay(chan)
# 				if k == len(seq)-1:
# 					post_delay = self.get_post_delay(chan)

# 				if pre_delay!=0 or post_delay!=0:
# 					rep = segment_play_info[1]
# 					segment_play_info[1] = 1
# 					input_data = self.generate_input_data(segment_play_info, chan, pre_delay, post_delay)
# 					sequence_data_single_channel.append(input_data)

# 					# If only one, go to next segment in the sequence.
# 					if rep == 1 :
# 						continue
# 					else:
# 						segment_play_info[1] = rep -1

# 				sequence_data_single_channel.append(self.generate_input_data(segment_play_info, chan))

# 			upload_data[chan] = sequence_data_single_channel

# 		return upload_data


# 	def generate_input_data(self, segment_play_info, channel, pre_delay=0, post_delay=0):
# 		'''
# 		function that will generate a dict that defines the input data, this will contain all the neccesary info to upload the segment.
# 		returns:
# 			dict with sequence info for a cerain channel (for parameters see the code).
# 		'''
# 		input_data = {'segment': segment_play_info[0], 
# 						'segment_name': self.make_segment_name(segment_play_info[0], pre_delay, post_delay),
# 						'ntimes': segment_play_info[1],
# 						'prescaler': segment_play_info[2],
# 						'pre_delay': pre_delay,
# 						'post_delay': post_delay}
# 		unique = getattr(self.segment_bin.get_segment(segment_play_info[0]), channel).unique
# 		input_data['unique'] = unique
# 		# Make unique uuid's for each segment
# 		if unique == True:
# 			input_data['identifier'] = [uuid.uuid4() for i in range(segment_play_info[1])]

# 		return input_data

# 	def make_segment_name(self, segment, pre_delay, post_delay):
# 		'''
# 		function that makes the name of the segment that is delayed.
# 		Note that if the delay is 0 there should be no new segment name.
# 		'''
# 		segment_name = segment
		
# 		if pre_delay!=0 or post_delay!= 0:
# 			segment_name = segment + '_' + str(pre_delay) + '_' + str(post_delay)

# 		return segment_name

		