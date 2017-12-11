import numpy as np
import datetime
from copy import deepcopy

import qcodes.instrument_drivers.Keysight.SD_common.SD_AWG as keysight_awg
import qcodes.instrument_drivers.Keysight.SD_common.SD_DIG as keysight_dig

class keysight_AWG():
	def __init__(self, segment_bin, channel_locations,channels):
		self.awg = dict()
		self.awg_memory = dict()
		# dict containing the number of the last segment number.
		self.segment_count = dict()
		self.current_waveform_number = 0

		self.channel_locations = channel_locations
		self.channels = channels

		self.vpp_max = 3 #Volt

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
		# step 1 collect vmin and vmax data, check if segments are intialized  (e.g. have at least one pulse):
		for i in sequence_data_raw:
			segment_name = i[0]
			if self.segment_bin.used(segment_name) == False:
				raise ValueError("Empty segment provided .. (segment name: '{}')".format(segment_name))
			v_min_max = self.segment_bin.get_segment(segment_name).Vmin_max_data
			self.adjust_vmin_vmax_data(v_min_max)

		# step 2 calculate Vpp/Voff needed for each channel + assign the voltages to each channel.
		self.adjust_vpp_data()

		# step 3 check memory allocation (e.g. what can we reuse of the sequences in the memory of the AWG.)
		mem_needed = dict()
		for i in self.awg:
			mem_needed[i] = 0

		for chan, sequence_data in sequence_data_processed.items():
			# loop trough all elements in the sequence and check how much data is needed.
			t = 0
			for i in sequence_data:
				segment_name = i['segment']
				repetitions= i['ntimes']
				unique = i['unique']

				# Check if stuff in the memory, if present, needs to be updated.
				if segment_name in self.segmentdata[chan] and unique == False:
					if self.segment_bin.get_segment(segment_name).last_mod <= self.segmentdata[chan][segment_name]['last_edit']:
						continue
				
				if unique == True:
					mem_needed[self.channel_locations[chan][0]] +=  self.segment_bin.get_segment(segment_name).total_time * repetitions
				else:
					mem_needed[self.channel_locations[chan][0]] +=  self.segment_bin.get_segment(segment_name).total_time


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
				segment_name = my_segment['segment']
				repetitions= my_segment['ntimes']
				unique = my_segment['unique']
				# Check if we need to skip the upload.
				if segment_name in self.segmentdata[chan] and unique == False:
					if self.segment_bin.get_segment(segment_name).last_mod <= self.segmentdata[chan][segment_name]['last_edit']:
						continue

				if unique == False:
					points = self.get_and_upload_waveform(chan,segment_name, time)
					time += points
				else:
					for uuid in range(repetitions):
						# my_segment['identifier'] = list with unique id's
						points = self.get_and_upload_waveform(chan,segment_name, time, my_segment['identifier'][uuid])
						time += points

		# step 5 make the queue in the AWG.
		self.flush_queues(self)

		for chan, sequence_data in sequence_data_processed.items():
			# get relevant awg and channel ()
			awg, awg_channels = self.channel_locations[chan]

			awg_name = self.channel_locations[chan][0]
			awg[awg_name]

			# First element needs to have the PXI trigger.
			first_element = True

			for segmentdata in sequence_data:
				if segmentdata['unique'] == True:
					for uuid in segmentdata['identifier']:
						awg_number = self.channel_locations[chan][1]
						seg_num = self.segmentdata[chan][uuid]
						if first_element ==  True:
							trigger_mode = 2
							first_element = False
						else : 
							trigger_mode = 0
						start_delay = 0
						cycles = 1
						prescaler = 0
						awg[awg_name].awg_queue_waveform(awg_number,seg_num,trigger_mode,start_delay,cycles,prescaler)

				else:
					awg_number = self.channel_locations[chan][1]
					seg_num = self.segmentdata[chan][segmentdata['segment']]
					if first_element ==  True:
						trigger_mode = 2
						first_element = False
					else :
						trigger_mode = 0

					start_delay = 0
					cycles = segmentdata['ntimes']
					prescaler = 0
					awg[awg_name].awg_queue_waveform(awg_number,seg_num,trigger_mode,start_delay,cycles,prescaler)


	def get_and_upload_waveform(self, channel, segment_name, time, uuid=None):
		'''
		get the wavform for channel with the name segment_name.
		The waveform occurs at time time in the sequence. 
		This function also adds the waveform to segmentdata variable
		'''
		seg_number = self.get_new_segment_number(channel)
		segment_data = self.segment_bin.get_segment(segment_name).get_waveform(channel, self.vpp_data, time, np.float32)

		wfv = keysight_awg.SD_AWG.new_waveform_from_double(seg_number, segment_data)
		
		awg_name = self.channel_locations[chan][0]
		awg[awg_name].load_waveform(wfv, seg_number)

		last_mod = self.segment_bin.get_segment(segment_name).last_mod
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
			
			if self.vpp_data[self.channels[0]]['v_pp'] is not None:
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
		print(self.awg[awg_name])
		self.awg[awg_name].set_channel_amplitude(new_vpp,chan_number)
		self.awg[awg_name].set_channel_offset(new_data['v_off'],chan_number)

		target['v_pp'] = new_vpp
		target['v_off']= new_data['v_off']

	def start(self):
		'''
		Function to apply the set the right triggering for the keysight AWG units.
		Triggering is done via the PXI triggers to make sure the that the system works correctly.
		'''

		# use a awg to send the trigger (does not matter which one)(here the first defined channel)
		awg_name = self.channel_locations[self.channels[0]][0]
		self.awg[awg_name].awg.PXItriggerWrite(4000,0)

		# set up the right trigger config
		for i in awg.items:
			i[1].awg_config_external_trigger(1,4000,1)
			i[1].awg_start(1)

		# trigger the system.
		# this will keel the waveform playing in a contineous mode.
		self.awg[awg_name].awg.PXItriggerWrite(4000,1)

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
			i[1].flush_waveform()

		for i in self.segment_count.items():
			i[1] = 0

		for i in self.awg_memory:
			i = self.maxmem
		print("Done.")

	def flush_queues(self):
		'''
		Remove all the queues form the channels in use.
		'''
		for i in self.channel_locations:
			self.awg[i[0]].awg_flush(i[1])

	def set_channel_properties(self):
		'''
		Sets how the channels should behave e.g., for now only arbitrary wave implemented.
		'''

		for i in self.channel_locations:
			# 6 is the magic number of the arbitary waveform shape.
			self.awg[i[0]].set_channel_wave_shape(6,i[1])