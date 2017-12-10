import numpy as np
import datetime
from copy import deepcopy


class keysight_awg():
	def __init__(self, segment_bin, channel_locations,channels):
		self.awg = dict()
		self.awg_memory = dict()
		self.current_waveform_number = 0

		self.channel_locations = channel_locations
		self.channels = channels

		self.voff_max = 1.5 # Volt (plus minus)
		self.vpp_max = 3 #Volt

		# setting for the amount of voltage you can be off from the optimal setting for a channels
		# e.g. when you are suppose to input
		self.voltage_tolerance = 0.2
		# data struct that contains the basic information for where in the memory of the awg which segment can be found.
		# Note that this might change when inplementing advanved looping.
		self.segmentdata = dict()
		for i in self.channels:
			self.segmentdata[i] = dict() #name segment + location

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
		for i in self.awg_memory:
			if alloc > i:
				allow = i
		return alloc

	def upload(self, sequence_data):
		# step 1 collect vmin and vmax data, check if segments are intialized  (e.g. have at least one pulse):
		for i in sequence_data:
			segment_name = i[0]
			if self.segment_bin.used(segment_name) == False:
				raise ValueError("Empty segment provided .. (segment name: '{}')".format(segment_name))
			v_min_max = self.segment_bin.get_segment(segment_name).Vmin_max_data
			self.adjust_vmin_vmax_data(v_min_max)

		# step 2 calculate Vpp/Voff needed for each channel.
		self.adjust_vpp_data()

		# step 3 check memory allocation (e.g. what can we reuse of the old sequences)
		mem_needed = 0
		for i in sequence_data:
			segment_name = i[0]
			if segment_name in self.segmentdata:
				if self.segment_bin.get_segment(segment_name).last_mod > self.segmentdata(segment_name):
					print('discarded , ', segment_name)
					continue
			else:
				mem_needed +=  self.segment_bin.get_segment(segment_name).total_time
		print(mem_needed)

		# If memory full, clear
		if mem_needed > self.allocatable_mem:
			print("memory cleared .. upload will take a bit longer.")
			self.clear_mem()

		# step 4 upload the sequences to the awg.
		for i in sequence_data:
			segment_name = i[0]
			if segment_name in self.segmentdata:
				if self.segment_bin.get_segment(segment_name).last_mod <= self.segmentdata(segment_name):
					segmend_data, channels = self.segment_bin.get_pulse(segment_name)
					# This is the location to do post processing?
	
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
			# Check if voltages are physical
			if vmax-vmin > self.vpp_max:
				raise ValueError("input range not supported (voltage peak to peak of {} detected)".format(vmax-vmin))
			if vmax > self.vpp_max + self.voff_max:
				raise ValueError("input range not supported (voltage of %d V detected) (max {} V)".format(vmax, self.vpp_max + self.voff_max))
			if vmin < - self.vpp_max- self.voff_max:
				raise ValueError("input range not supported (voltage of %d V detected) (min {} V)".format(vmin, -self.vpp_max - self.voff_max))

			# check if current settings of the awg are fine.
			if self.vpp_data[self.channels[0]]['v_pp'] is not None:
				vpp_current  = self.vpp_data[i]['v_pp']
				voff_current = self.vpp_data[i]['v_off']

				vmin_current = voff_current - vpp_current/2
				vmax_current = voff_current + vpp_current/2

				if vmin_current > vmin or vmax_current < vmax:
					voltage_range_reset_needed = True
				# note if the voltages needed are significantly smaller, we also want to do a voltage reset.
				if vmin_current*(1-self.voltage_tolerance) < vmin or vmax_current*(1-self.voltage_tolerance) > vmax:
					voltage_range_reset_needed = True

			# convert to peak to peak and offset voltage.
			vpp_test[i]['v_pp'] = vmax - vmin
			vpp_test[i]['v_off']= (vmax + vmin)/2

		# 2) if vpp fals not in old specs, clear memory and add new ranges.
		if self.vpp_data[self.channels[0]]['v_pp'] is not None or voltage_range_reset_needed == True:
			self.clear_mem()

			for i in self.channels:
				self.update_vpp_single(vpp_test[i],self.vpp_data[i])

	def update_vpp_single(self, new_data, target):
		'''
		updata the voltages, with the tolerance build in.
		'''

		new_vpp = new_data['v_pp'] * (1 + self.voltage_tolerance)
		if new_vpp > self.vpp_max:
			new_vpp = self.vpp_max

		target['v_pp'] = new_vpp
		target['v_off']= new_data['v_off']

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
		'''
		add an awg to tge keysight object. As awg a qcodes object from the keysight driver is expected.
		name is name you want to give to your awg object. This needs to be unique and should is to describe which
		channel belongs to which awg.
		'''
		self.awg[name] = awg, self.maxmem
		self.awg_memory[name] =self.maxmem
		self.segdata_awg_mem[name] = dict()

	def clear_mem(self):
		'''
		Clears ram of all the AWG's
		Clears segment_loc on pc. (TODO)
		'''
		print("AWG memory is being cleared.")
		self.segmentdata = dict()
		for i in self.awg.items():
			i[1].flush_waveform()
		for i in self.awg_memory:
			i = self.maxmem
		print("Done.")