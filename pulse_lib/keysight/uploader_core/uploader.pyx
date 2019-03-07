from libcpp cimport bool
from libc.stdlib cimport malloc, free
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.pair cimport pair
from libcpp.map cimport map as mapcpp
from cython.operator import dereference, postincrement

import numpy as np
cimport numpy as np


cdef extern from "keysight_awg_post_processing_and_upload.h":
	struct waveform_raw_upload_data:
		vector[double*] *wvf_data
		# npt added here already as you cannot fetch it without the gil 
		vector[int] *wvf_npt
		pair[double, double] *min_max_voltage
		vector[double*] *DSP_param
		short *upload_data
		int *npt
		vector[int] data_location_on_AWG

ctypedef waveform_raw_upload_data* waveform_raw_upload_data_ptr

cdef extern from "keysight_awg_post_processing_and_upload.h":

	cdef cppclass cpp_uploader:
		cpp_uploader() except +
		void add_awg_module(string name, int chassis, int slot) nogil
		void add_upload_job(mapcpp[string, mapcpp[int, waveform_raw_upload_data_ptr]] *upload_data) nogil
		void release_memory(mapcpp[string, mapcpp[int, waveform_raw_upload_data_ptr]] *upload_data) nogil
		void resegment_memory() nogil

cdef struct s_waveform_info:
	pair[double, double] min_max_voltage
	double integral

ctypedef s_waveform_info waveform_info

cdef class keysight_upload_module():
	"""
	This is a speraturate module for the upload in keysight units. This module also does some last post processing on the waveform (e.g. DSP/convert to short/extend them, so they fit into the memory ...)
	The upload in writtin in C, so can be fully run without gil in a multhithreaded way. 
	This module is in now way a replacement for the python module and has as main function to get the waveform into the memory
	"""
	cdef cpp_uploader *keysight_uploader

	def __cinit__(self):
		self.keysight_uploader = new cpp_uploader()
		cdef mapcpp[string, pair[string, int]] channel_to_AWG_map
	def add_awg_module(self, name, module):
		'''
		add an AWG module to the keysight object.
		Args:
			module (qCodeS driver) : qcodes object of the AWG
		'''
		cdef string c_name = name.encode('utf8')
		print(name, c_name)

		self.keysight_uploader.add_awg_module(c_name, module.chassis, module.slot)

	def add_upload_data(self, waveform_cache_container waveform_cache):
		cdef mapcpp[string, mapcpp[int, waveform_raw_upload_data_ptr]] *AWG_raw_upload_data
		AWG_raw_upload_data = &waveform_cache.AWG_raw_upload_data

		with nogil:
			self.keysight_uploader.add_upload_job(AWG_raw_upload_data)

		AWG_init_data = dict()


		cdef mapcpp[string, pair[string, int]].iterator it = waveform_cache.channel_to_AWG_map.begin()
		cdef waveform_raw_upload_data_ptr channel_data
		while(it != waveform_cache.channel_to_AWG_map.end()):
			# make tuple with gate voltages for the channel and location of the AWG memeory where the waveforms are stored. 

			channel_data = dereference(dereference(AWG_raw_upload_data.find(dereference(it).second.first)).second.find(dereference(it).second.second)).second

			min_max_voltage = (channel_data.min_max_voltage.first, channel_data.min_max_voltage.second,)

			# (memory_upload_location, cycles, prescalor) (last two not implented in c++ upload part)
			upload_locations = list( (channel_data.data_location_on_AWG, 1, 0))

			AWG_init_data[dereference(it).first] = (min_max_voltage, upload_locations)

			postincrement(it)

		return AWG_init_data

	def resegment_memory(self):
		'''
		Apply a full re-segmentation of the memory of the AWG.
		Means a full flush and also remaking of the memory table (might take a while...)
		'''
		self.keysight_uploader.resegment_memory()

	cdef release_memory(self, mapcpp[string, mapcpp[int, waveform_raw_upload_data_ptr]] *AWG_raw_upload_data):
		self.keysight_uploader.release_memory(AWG_raw_upload_data)



cdef class waveform_cache_container():
	cdef mapcpp[string, pair[string, int]] channel_to_AWG_map
	cdef mapcpp[string, mapcpp[int, waveform_raw_upload_data_ptr]] AWG_raw_upload_data
	cdef dict waveform_chache_python

	def __init__(self, channel_to_AWG_map_py, voltage_limits):
		'''
		Initialize the waveform cache.
		Args:
			channel_to_AWG_map_py (dict<str channel_name, tuple<str AWG_name, int channel_number>)
			voltage_limits (dict<str channel_name, typle<double min_voltage, double max_voltage>)
		'''
		self.waveform_chache_python = dict()

		for key, value in channel_to_AWG_map_py.items():
			new_value = (value[0].encode(), value[1])
			self.channel_to_AWG_map[key.encode()] = new_value

			waveform_cache = waveform_upload_chache(voltage_limits[key])
			self.waveform_chache_python[key] = waveform_cache
			self.AWG_raw_upload_data[new_value[0]][new_value[1]] = waveform_cache.return_raw_data()

	def __getitem__(self, str key):
		'''
		get waveform_upload_chache object
		Args:
			key (str) : name of the channel
		'''
		return self.waveform_chache_python[key]

	@property
	def npt(self):
		'''
		Return the number of point that is saved in the caches (= total number of points that need to be uploaded).

		Note that it is assumed that you run this function when all the caches have been populated and have similar size.
		If you want to know npt per chache, you should call self['channel_name'].npt
		'''
		if len(self.waveform_chache_python) == 0 :
			raise ValueError("No waveforms presents in waveform chache container ...")
		
		# get first key of chache object (the pyton one)
		idx = next(iter(self.waveform_chache_python))
		return self[idx].npt

	def generate_DC_compenstation(self):
		'''
		generate a DC compensation of the pulse.
		As assuallly we put condensaters in between the AWG and the gate on the sample, you need to correct for the fact that the low fequencies are not present in your transfer function.
		This can be done simply by making the total integral of your function 0.
		'''

		cdef int compensation_time = 0
		cdef int wvf_npt = 0
		for chan in self.waveform_chache_python:
			if self[chan].compensation_time > compensation_time:
				compensation_time = self[chan].compensation_time 
				wvf_npt = self[chan].npt
		# make sure we have modulo 10 time
		cdef int total_pt = compensation_time + wvf_npt
		cdef int mod = total_pt%10
		if mod != 0:
			total_pt += 10-mod
		compensation_time = total_pt - wvf_npt

		#generate the compensation
		for chan in self.waveform_chache_python:
			self[chan].generate_voltage_compensation(compensation_time)

cdef class waveform_upload_chache():
	"""object that holds some cache for uploads and does some basic calculations"""
	cdef waveform_raw_upload_data raw_data
	cdef vector[waveform_info] wvf_info
	cdef vector[double *] wvf_data
	cdef vector[int] wvf_npt
	cdef pair[double, double] min_max_voltage
	cdef pair[double, double] compenstation_limit
	cdef int _npt
	cdef list data

	def __init__(self, compenstation_limit):
		self.compenstation_limit = compenstation_limit
		self._npt = 0
		self.data = []

	def add_data(self, np.ndarray[np.double_t,ndim=1] wvf, v_min_max, integral):
		'''
		wvf (np.ndarray[ndim = 1, dtype=double]) : waveform
		v_min_max (tuple) : maximum/minimum voltage of the current segment
		integral (double) : integrated value of the waveform
		'''
		cdef waveform_info data_info

		if self._npt == 0:
			self.min_max_voltage = v_min_max
		else:
			if self.min_max_voltage.first > v_min_max[0]:
				self.min_max_voltage.first = v_min_max[0]
			if self.min_max_voltage.second < v_min_max[1]:
				self.min_max_voltage.second = v_min_max[1]

		self._npt += wvf.size
		data_info.min_max_voltage = v_min_max
		data_info.integral = integral

		self.wvf_info.push_back(data_info)
		cdef double* my_data =  <double*> wvf.data
		self.wvf_data.push_back(my_data)
		# use dummy python list for reference counting ..
		self.data.append(wvf)

		self.wvf_npt.push_back(wvf.size)

	@property
	def integral(self):
		cdef double integral = 0
		for i in self.wvf_info:
			integral += i.integral
		return integral

	@property
	def compensation_time(self):
		'''
		return the minimal compensation time that is needed.
		Returns:
			compensation_time : minimal duration that is needed for the voltage compensation
		'''
		if self.compenstation_limit.first == 0 or self.compenstation_limit.second == 0:
			return 0

		cdef double comp_time = self.integral
		if comp_time <= 0:
			return -comp_time / self.compenstation_limit.second
		else:
			return -comp_time / self.compenstation_limit.first

	@property
	def npt(self):
		return self._npt

	cpdef void generate_voltage_compensation(self, double time):
		'''
		make a voltage compenstation pulse of time t
		Args:
			time (double) : time of the compenstation in ns
		'''
		cdef double voltage = 0
		if round(time) == 0:
			voltage = 0
		else:
			voltage = self.integral/round(time)

		self.add_data(np.full((int(round(time)),), voltage), (voltage, voltage), -self.integral)

	cdef waveform_raw_upload_data* return_raw_data(self):
		self.raw_data.wvf_data = &self.wvf_data
		self.raw_data.wvf_npt = &self.wvf_npt
		self.raw_data.min_max_voltage = &self.min_max_voltage
		self.raw_data.npt = &self._npt

		return &self.raw_data
