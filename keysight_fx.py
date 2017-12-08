import numpy as np
import datetime

class keysight_awg():
	def __init__(self, segment_bin, channel_locations,channels):
		self.awg = dict()
		self.awg_memory = dict()
		self.current_waveform_number = 0

		self.channel_locations = channel_locations
		self.channels = channels
		self.segmentdata = dict()
		self.segdata_awg_mem = dict()
		self.vpp_data = dict()
		for i in self.channels:
			self.vpp_data[i] = {"V_min" : None, "V_max" : None}
		self.segment_bin = segment_bin
		print(self.vpp_data)
		self.maxmem = 1e9
	def upload(self, sequence_data):
		# step 1 prepare for upload, check how much memory is needed. (and is available)
		mem_needed = 0
		for i in sequence_data:
			segment_name = i[0]
			if segment_name in self.segmentdata:
				if self.segment_bin.get_segment(segment_name).latest_mod > self.segmentdata(segment_name):
					continue
			else:
				mem_needed +=  self.segment_bin.get_segment(segment_name).total_time
		

		# If memory full, clear
		if mem_needed > self.allocatable_mem:
			self.clear_mem()

		# step 2 upload
		for i in sequence_data:
			segment_name = i[0]
			if segment_name not in self.segmentdata:
				if self.segment_bin.get_segment(segment_name).last_edit <= self.segmentdata(segment_name):
					segmend_data, channels = segment_bin.get_pulse(segment_name)
					# This is the location to do post processing?
				

	def start(self, sequence_data):
		return


		# mem_needed
		# for i in 
		# Ask segmentbin to check if elements are present, if not -- upload
		# self.segment_bin.upload('INIT')
		# Upload the relevant segments.

		# if self.usable_mem - len(wfv) < 0:
		# 	raise Exception("AWG Full :(. Clear all the ram... Note that this error should normally be prevented automatically.")

		# wfv.astype(np.int16)
		return


	def check_mem_availability(self, num_pt):
		return True

	def add_awg(self, name, awg):
		self.awg[name] = awg, self.maxmem
		self.awg_memory[name] =self.maxmem
		self.segdata_awg_mem[name] = dict()
	@property
	def allocatable_mem(self):
		alloc = self.maxmem
		for i in self.awg_memory:
			if alloc > i:
				allow = i
		return alloc


	def clear_mem(self):
		'''
		Clears ram of all the AWG's
		'''
		self.segmentdata = dict()
		for i in self.awg():
			i.flush_waveform()
		for i in self.awg_memory:
			i = self.maxmem