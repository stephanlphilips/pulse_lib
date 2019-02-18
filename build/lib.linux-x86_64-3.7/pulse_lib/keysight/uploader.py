import threading as th
import numpy as np
import time
from pulse_lib.keysight.uploader_core.uploader import waveform_cache_container,waveform_upload_chache
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
	def __init__(self, AWGs, cpp_uploader,channel_names, channel_locations, channel_delays, channel_compenstation_limits):
		'''
		Initialize the keysight uploader. 
		Args:
			AWGs (dict<awg_name,QcodesIntrument>) : list with AWG's
			cpp_uploader (keysight_upload_module) : class that performs normalisation and conversion of the wavorm to short + upload.
			channel_names(list) : list with all the names of the channels
			channel_locations (dict): dict with channel and AWG+channel location
			channel_compenstation_limits (dict) : dict with channel name as key and tuple as value with lower and upper limit
		Returns:
			None
		'''
		self.memory_allocation = dict()
		self.upload = True
		# TODO reinit memory on start-up
		self.AWGs = AWGs
		self.cpp_uploader =cpp_uploader
		self.channel_names = channel_names
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
		4a) convert data in an aprropriate upload format (c++)
		4b) upload all data (c++)
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
			waveform_cache = waveform_cache_container(self.channel_map, self.channel_compenstation_limits)
			

			pre_delay = 0
			post_delay = 0
			print("\n\nnew call on pulse data\n")
			for i in range(len(job.sequence)):

				seg = job.sequence[i][0]
				n_rep = job.sequence[i][1]
				prescaler = job.sequence[i][2]
				
				# TODO add precaler in as sample rate
				for channel in self.channel_names:
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
			waveform_cache.generate_DC_compenstation()
			
			end2 = time.time()
			
			# 3) DSP correction
			# TODO later

			# 4a+b)
			job.upload_data = self.cpp_uploader.add_upload_data(waveform_cache)

			# submit the current job as completed.
			self.upload_done.append(job)

			end3 = time.time()
			print("time needed to render and compenstate",end3 - start)
			print("rendering = ", end1 - start)
			print("compensation = ", end2 - end1)
			print("cpp conversion to short = ", end3 - end2)


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
		self.upload_data = None
	
	def add_dsp_function(self, DSP):
		self.DSP =True
		self.DSP_func = DSP


