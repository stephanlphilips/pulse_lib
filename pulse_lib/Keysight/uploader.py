import threading as th
import numpy as np

from pulse_lib.Keysight.AWG_memory_manager import Memory_manager

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
	def __init__(self, AWGs, channel_locations, channel_delays):
		'''
		Initialize the keysight uploader. 
		Args:
			AWGs (list<QcodesIntrument>) : list with AWG's
			channel_locations (dict): dict with channel and AWG+channel location
		Returns:
			None
		'''
		self.memory_allocation = dict()

		# TODO reinit memory on start-up
		for i in AWG:
			self.memory_allocation[i.name]= Memory_manager()
		self.channel_map = channel_locations
		self.channel_delays = channel_delays

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
	# @mk_thread
	def uploader(self):
		'''
		Class taking care of putting the waveform on the right AWG. This is a continuous thread that is run in the background.

		Steps:
		1) get all the upload data
		2) perform DC correction (if needed)
		3) perform DSP correction (if needed)
		4) upload all data
		5) write in the job object the resulting locations of sequences that have been uploaded.

		'''

		job = self.get_new_job(self)

		upload_data = dict()

class upload_job(object):
	"""docstring for upload_job"""
	def __init__(self, sequence, index=0, neutralize=True, priority=0):
		self.sequence = sequence
		self.index = index
		self.neutralize = True
		self.priority = priority
		self.DSP = False
	
	def add_dsp_function(self, DSP):
		self.DSP =True
		self.DSP_func = DSP

	def assign_upload_locations(self):
		pass
		
		

