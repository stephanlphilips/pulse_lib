import numpy as np
import datetime
from copy import deepcopy, copy
import matplotlib.pyplot as plt
import qcodes.instrument_drivers.Keysight.SD_common.SD_AWG as keysight_awg
import qcodes.instrument_drivers.Keysight.SD_common.SD_DIG as keysight_dig

import sys
sys.path.append("C:/Program Files (x86)/Keysight/SD1/Libraries/Python/")
import keysightSD1


class keysight_AWG():
	def __init__(self, segment_bin, channel_locations,channels, channel_delays):
		self.awg = dict()
		self.awg_memory = dict()
		# dict containing the number of the last segment number.
		self.segment_count = dict()
		self.current_waveform_number = 0

		self.channel_locations = channel_locations
		self.channels = channels
		self.channel_delays = channel_delays

		self.vpp_max = 3 #Volt

		# init HVI object
		self.HVI = keysightSD1.SD_HVI()

		# setting for the amount of voltage you can be off from the optimal setting for a channels
		# e.g. when you are suppose to input
		self.voltage_tolerance = 0.2
		# data struct that contains the basic information for where in the memory of the awg which segment can be found.
		# Note that this might change when inplementing advanved looping.
		self.segmentdata = dict()
		for i in self.channels:
			self.segmentdata[i] = dict() #name segment + location and date of construction of segment

		self.vpp_data = dict()
		for i in self.channels:
			self.vpp_data[i] = {"v_pp" : None, "v_off" : None}

		self.v_min_max_combined = dict()
		for i in self.channels:
			self.v_min_max_combined[i] = {"v_min" : None, "v_max" : None}

		self.segment_bin = segment_bin
		
		self.maxmem = 1e9

		# General data
		self.n_rep = 0 # number of times to repeat the sequence (0 is infinite).
		self.length_sequence = 1

	@property
	def allocatable_mem(self):
		alloc = self.maxmem
		for i in self.awg_memory.items():
			if alloc > i[1]:
				allow = i[1]
		return alloc

	def get_new_segment_number(self, channel):
		'''
		gets a segment number for the new segment. These numbers just need to be unique.
		'''
		awg_name = self.channel_locations[channel][0]
		self.segment_count[awg_name] += 1
		if self.segment_count[awg_name] > 2000 :
			print("number of segments on the awg (",self.segment_count[awg_name], ")is greater than 2000, this might cause problems?")
		return self.segment_count[awg_name]

	def upload(self, sequence_data_raw, sequence_data_processed):
		# TODO put at better location later
		self.set_channel_properties()
		self.flush_queues()
		# step 1 collect vmin and vmax data, check if segments are intialized  (e.g. have at least one pulse):
		for i in sequence_data_raw:
			segment_id = i[0]
			if self.segment_bin.used(segment_id) == False:
				raise ValueError("Empty segment provided .. (segment name: '{}')".format(segment_id))
			v_min_max = self.segment_bin.get_segment(segment_id).Vmin_max_data
			self.adjust_vmin_vmax_data(v_min_max)

		# step 2 calculate Vpp/Voff needed for each channel + assign the voltages to each channel.
		self.adjust_vpp_data()

		# step 3 check memory allocation (e.g. what can we reuse of the sequences in the memory of the AWG.)
		tot_channel_delay = self.calculate_total_channel_delay()
		mem_needed = dict()
		for i in self.awg:
			mem_needed[i] = 0

		for chan, sequence_data in sequence_data_processed.items():
			
			# Check if segments can be reused as expected (if channell has a delay, first and segment cannot be repeated. )
			for i in sequence_data:
				segment_id = i['segment']
				segment_name = i['segment_name']
				repetitions= i['ntimes']
				unique = i['unique']
				pre_delay= i['pre_delay']
				post_delay = i['post_delay']

				# Check if the segment is in the memory and still up to date (if not, go on.)
				if self.segment_in_mem(segment_id, segment_name, chan) and unique == False:
					continue
				
				if unique == True:
					mem_needed[self.channel_locations[chan][0]] += pre_delay + post_delay + self.segment_bin.get_segment(segment_id).total_time * repetitions
				else:
					mem_needed[self.channel_locations[chan][0]] += pre_delay + post_delay + self.segment_bin.get_segment(segment_id).total_time


		# for i in self.awg:
		# 	print("memory needed for awg {} is {} points.".format(i,mem_needed[i]))


		# If memory full, clear (if one is full it is very likely all others are also full, so we will just clear everything.)
		for i in self.awg:
			if mem_needed[i] > self.allocatable_mem:
				print("memory cleared .. upload will take a bit longer.")
				self.clear_mem()

		# step 4 upload the sequences to the awg.
		for chan, sequence_data in sequence_data_processed.items():
			# Upload here sequences.

			# Keep counting time of the segments. This is important for IQ data.
			time = 0
			for my_segment in sequence_data:
				segment_id = my_segment['segment']
				segment_name = my_segment['segment_name']
				repetitions= my_segment['ntimes']
				unique = my_segment['unique']
				pre_delay= my_segment['pre_delay']
				post_delay = my_segment['post_delay']
				
				# Check if the segment is in the memory and still up to date (if not, go on.)
				if self.segment_in_mem(segment_id, segment_name, chan) and unique == False:
					continue

				if unique == False:
					points = self.get_and_upload_waveform(chan,my_segment, time)

					time += points*repetitions
				else:
					for uuid in range(repetitions):
						# my_segment['identifier'] = list with unique id's
						points = self.get_and_upload_waveform(chan,my_segment, time, my_segment['identifier'][uuid])

						time += points
			self.length_sequence = time

		# step 5 make the queue in the AWG.
		for chan, sequence_data in sequence_data_processed.items():
			# get relevant awg
			awg_name = self.channel_locations[chan][0]
			awg_number = self.channel_locations[chan][1]

			# First element needs to have the HVI trigger.
			first_element = True
			for segmentdata in sequence_data:
				if segmentdata['unique'] == True:
					for uuid in segmentdata['identifier']:
						seg_num = self.segmentdata[chan][uuid]['mem_pointer']
						if first_element ==  True:
							trigger_mode = 1
							first_element = False
						else : 
							trigger_mode = 0
						start_delay = 0
						cycles = 1
						prescaler = segmentdata['prescaler']
						self.awg[awg_name].awg_queue_waveform(awg_number,seg_num,trigger_mode,start_delay,cycles,prescaler)

				else:
					seg_num = self.segmentdata[chan][segmentdata['segment_name']]['mem_pointer']
					if first_element ==  True:
						trigger_mode = 1
						first_element = False
					else :
						trigger_mode = 0

					start_delay = 0
					cycles = segmentdata['ntimes']
					prescaler = segmentdata['prescaler']
					self.awg[awg_name].awg_queue_waveform(awg_number,seg_num,trigger_mode,start_delay,cycles,prescaler)

	def segment_in_mem(self, seg_id, seg_name, channel):
		'''
		function that checks is certain segment in already present in the memory of the awg
		input:
			1 list item from a sequence element.
		Returns:
			True/False
		'''

		if seg_name in self.segmentdata[channel]:
			if self.segment_bin.get_segment(seg_id).last_mod <= self.segmentdata[channel][seg_name]['last_edit']:
				return True

		return False

	def get_and_upload_waveform(self, channel, segment_info, time, uuid=None):
		'''
		get the wavform for channel with the name segment_name.
		The waveform occurs at time time in the sequence. 
		This function also adds the waveform to segmentdata variable
		'''

		segment_id = segment_info['segment']
		segment_name = segment_info['segment_name']
		pre_delay= segment_info['pre_delay']
		post_delay = segment_info['post_delay']

		# point data of the segment (array to be uploaded).
		segment_data = self.segment_bin.get_segment(segment_id).get_waveform(channel, self.vpp_data, time, pre_delay, post_delay, np.float32)

		wfv = keysight_awg.SD_AWG.new_waveform_from_double(0, segment_data)
		awg_name = self.channel_locations[channel][0]

		seg_number = self.get_new_segment_number(channel)
		self.awg[awg_name].load_waveform(wfv, seg_number)
		# print("plotting {}, {}".format(channel, segment_id))
		# plt.plot(segment_data)
		last_mod = self.segment_bin.get_segment(segment_id).last_mod
		# upload data
		if uuid is None:
			self.segmentdata[channel][segment_name] = dict()
			self.segmentdata[channel][segment_name]['mem_pointer'] = seg_number
			self.segmentdata[channel][segment_name]['last_edit'] = last_mod
		else:
			self.segmentdata[channel][uuid] = dict()
			self.segmentdata[channel][uuid]['mem_pointer'] = seg_number
			self.segmentdata[channel][uuid]['last_edit'] = last_mod
		return len(segment_data)

	def adjust_vmin_vmax_data(self, Vmin_max_data):
		'''
		Function that updates the values of the minimun and maximun volages needed for each channels.
		Input dict with for each channel max and min voltage for al segments that will be played in a sequence.
		'''
		for i in self.channels:
			if self.v_min_max_combined[i]['v_min'] is None:
				self.v_min_max_combined[i]['v_min'] = Vmin_max_data[i]['v_min']
				self.v_min_max_combined[i]['v_max'] = Vmin_max_data[i]['v_max']
				continue

			if self.v_min_max_combined[i]['v_min']  > Vmin_max_data[i]['v_min']:
				self.v_min_max_combined[i]['v_min'] = Vmin_max_data[i]['v_min']

			if self.v_min_max_combined[i]['v_max']  < Vmin_max_data[i]['v_max']:
				self.v_min_max_combined[i]['v_max'] = Vmin_max_data[i]['v_max']

	def adjust_vpp_data(self):
		'''
		Function that adjust the settings of the peak to peak voltages of the awg.
		Check if the sequence can be made with the current settings, if not, all the memory will be purged.
		The reason not only to purge the channels where it is needed is in the case of the use of virtual gates,
		since then all channels will need to be reuploaded anyway ..
		An option to manually enforce a vpp and voff might also be nice?
		'''

		# 1) generate vpp needed, and check if it falls in the range allowed.
		vpp_test = deepcopy(self.vpp_data)
		voltage_range_reset_needed = False

		for i in self.channels:
			vmin = self.v_min_max_combined[i]['v_min']
			vmax = self.v_min_max_combined[i]['v_max']
			# Check if voltages are physical -- note that the keysight its offset is kind of a not very proper defined parameter.
			if vmax > self.vpp_max/2:
				raise ValueError("input range not supported (voltage of {} V detected) (max {} V)".format(vmax, self.vpp_max/2))
			if vmin < - self.vpp_max/2:
				raise ValueError("input range not supported (voltage of {} V detected) (min {} V)".format(vmin, -self.vpp_max/2))

			# check if current settings of the awg are fine.
			if self.vpp_data[self.channels[0]]['v_pp'] is not None:
				vpp_current  = self.vpp_data[i]['v_pp']
				voff_current = self.vpp_data[i]['v_off']

				vmin_current = voff_current - vpp_current
				vmax_current = voff_current + vpp_current

				if vmin_current > vmin or vmax_current < vmax:
					voltage_range_reset_needed = True
				# note if the voltages needed are significantly smaller, we also want to do a voltage reset.
				if vmin_current*(1-2*self.voltage_tolerance) < vmin or vmax_current*(1-2*self.voltage_tolerance) > vmax:
					voltage_range_reset_needed = True

			# convert to peak to peak and offset voltage.
			vpp_test[i]['v_pp'] =(vmax - vmin)/2
			vpp_test[i]['v_off']= (vmax + vmin)/2

		# 2) if vpp fals not in old specs, clear memory and add new ranges.
		if self.vpp_data[self.channels[0]]['v_pp'] is None or voltage_range_reset_needed == True:
			self.clear_mem()

			for i in self.channels:
				self.update_vpp_single(vpp_test[i],self.vpp_data[i], i)

	def update_vpp_single(self, new_data, target, channel):
		'''
		Update the voltages, with the tolerance build in.
		'''
		new_vpp = new_data['v_pp'] * (1 + self.voltage_tolerance)
		if new_vpp > self.vpp_max/2:
			new_vpp = self.vpp_max/2
		awg_name = self.channel_locations[channel][0]
		chan_number = self.channel_locations[channel][1]

		self.awg[awg_name].set_channel_amplitude(new_vpp,chan_number)
		self.awg[awg_name].set_channel_offset(new_data['v_off'],chan_number)

		target['v_pp'] = new_vpp
		target['v_off']= new_data['v_off']

	def start(self):
		'''
		Function to apply the set the right triggering for the keysight AWG units.
		Triggering is done via the PXI triggers to make sure the that the system works correctly.
		'''

		# Launch the right HVI instance, set right parameters.
		self.HVI.stop()

		self.HVI.open("C:/V2_code/HVI/For_loop_single_sequence.HVI")

		self.HVI.assignHardwareWithIndexAndSlot(0,0,2)
		self.HVI.assignHardwareWithIndexAndSlot(1,0,3)
		self.HVI.assignHardwareWithIndexAndSlot(2,0,4)
		self.HVI.assignHardwareWithIndexAndSlot(3,0,5)

		# Length of the sequence
		self.HVI.writeIntegerConstantWithIndex(0, "length_sequence", int(self.length_sequence/10 + 1))
		self.HVI.writeIntegerConstantWithIndex(1, "length_sequence", int(self.length_sequence/10 + 1))
		self.HVI.writeIntegerConstantWithIndex(2, "length_sequence", int(self.length_sequence/10 + 1))
		self.HVI.writeIntegerConstantWithIndex(3, "length_sequence", int(self.length_sequence/10 + 1))


		# number of repetitions
		nrep = self.n_rep
		if nrep == 0:
			nrep = 1
		self.HVI.writeIntegerConstantWithIndex(0, "n_rep", nrep)
		self.HVI.writeIntegerConstantWithIndex(1, "n_rep", nrep)
		self.HVI.writeIntegerConstantWithIndex(2, "n_rep", nrep)
		self.HVI.writeIntegerConstantWithIndex(3, "n_rep", nrep)

		# Inifinite looping
		step = 1
		if self.n_rep == 0:
			step  = 0
		self.HVI.writeIntegerConstantWithIndex(0, "step", step)
		self.HVI.writeIntegerConstantWithIndex(1, "step", step)
		self.HVI.writeIntegerConstantWithIndex(2, "step", step)
		self.HVI.writeIntegerConstantWithIndex(3, "step", step)

		self.HVI.compile()
		self.HVI.load()
		self.HVI.start()

	def add_awg(self, name, awg):
		'''
		add an awg to tge keysight object. As awg a qcodes object from the keysight driver is expected.
		name is name you want to give to your awg object. This needs to be unique and should is to describe which
		channel belongs to which awg.
		'''
		# Make sure you start with a empty memory

		# awg.flush_waveform()
		self.awg[name] = awg
		self.awg_memory[name] =self.maxmem
		self.segment_count[name] = 0

	def clear_mem(self):
		'''
		Clears ram of all the AWG's
		Clears segment_loc on pc. (TODO)
		'''
		print("AWG memory is being cleared.")
		self.segmentdata = dict()
		for i in self.awg.items():
			i[1].flush_waveform(True)

		for awg, count in self.segment_count.items():
			count = 0

		self.segmentdata = dict()
		for i in self.channels:
			self.segmentdata[i] = dict() #name segment + location and date of construction of segment

		for i in self.awg_memory:
			i = self.maxmem
		print("Done.")

	def flush_queues(self):
		'''
		Remove all the queues form the channels in use.
		'''
		# awg2.awg_stop(1)
		print("all queue cleared")
		for channel, channel_loc in self.channel_locations.items():
			self.awg[channel_loc[0]].awg_stop(channel_loc[1])
			self.awg[channel_loc[0]].awg_flush(channel_loc[1])

	def set_channel_properties(self):
		'''
		Sets how the channels should behave e.g., for now only arbitrary wave implemented.
		'''
		print("channels set.")
		for channel, channel_loc in self.channel_locations.items():
			# 6 is the magic number of the arbitary waveform shape.
			self.awg[channel_loc[0]].set_channel_wave_shape(6,channel_loc[1])
			self.awg[channel_loc[0]].awg_queue_config(channel_loc[1], 1)

	def calculate_total_channel_delay(self):
		'''
		function for calculating how many ns time there is a delay in between the channels.
		Also support for negative delays...

		returns:
			tot_delay (the total delay)
		'''

		delays =  np.array( list(self.channel_delays.values()))
		tot_delay = np.max(delays) - np.min(delays)

		return tot_delay
