from libcpp cimport bool
from libc.stdlib cimport malloc, free
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.pair cimport pair
from libcpp.map cimport map as mapcpp
from cython.operator import dereference, postincrement
from libc.stdio cimport printf

import numpy as np
cimport numpy as np

cdef extern from "keysight_awg_post_processing_and_upload.h":
	cdef cppclass cpp_uploader:
		cpp_uploader() except +
		void add_awg_module(string name, string module_type, int chassis, int slot)
		void add_upload_job(mapcpp[string, mapcpp[int, waveform_raw_upload_data]] *upload_data)

	struct waveform_raw_upload_data:
		vector[double*] *wvf_data
		# np added here already as you cannot fetch it without the gil 
		vector[int] *wvf_npt
		pair[double, double] *min_max_voltage
		vector[double*] *DSP_param
		short *upload_data
		int npt



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

	def add_awg_module(self, module):
		'''
		add an AWG module to the keysight object.
		Args:
			module (qCodeS driver) : qcodes object of the AWG
		'''
		self.keysight_uploader.add_awg_module(module.name.encode(), module.type.encode(), module.chassis, module.slot)

	def add_upload_data(self, waveform_cache_container waveform_cache):
		cdef mapcpp[string, mapcpp[int, waveform_raw_upload_data]] *AWG_raw_upload_data
		AWG_raw_upload_data = &waveform_cache.AWG_raw_upload_data
		
		self.keysight_uploader.add_upload_job(AWG_raw_upload_data)

cdef class waveform_upload_chache():
	"""object that holds some cache for uploads and does some basic calculations"""
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

	cdef waveform_raw_upload_data return_raw_data(self):
		cdef waveform_raw_upload_data raw_data

		raw_data.wvf_data = &self.wvf_data
		raw_data.wvf_npt = &self.wvf_npt
		raw_data.min_max_voltage = &self.min_max_voltage
		raw_data.npt = self._npt

		return raw_data

cdef class waveform_cache_container():
	cdef mapcpp[string, pair[string, int]] channel_to_AWG_map
	cdef mapcpp[string, mapcpp[int, waveform_raw_upload_data]] AWG_raw_upload_data
	cdef dict waveform_chache_python

	def __init__(self, channel_to_AWG_map_py):

		for key, value in channel_to_AWG_map_py.items():
			new_value = (value[0].encode(), value[1])
			self.channel_to_AWG_map[key.encode()] = new_value

		self.waveform_chache_python = dict()

	def __setitem__(self, str key, waveform_upload_chache value):
		'''
		Assign waveform_upload_chache object to each channel
		Args:
			key (str): name of the channel
			value (waveform_upload_chache) : upload cache object
		'''
		cdef string _key = key.encode()

		cdef string awg_name = self.channel_to_AWG_map[_key].first
		cdef int channel_number = self.channel_to_AWG_map[_key].second

		self.AWG_raw_upload_data[awg_name][channel_number] = value.return_raw_data()
		self.waveform_chache_python[key] = value

	def __getitem__(self, str key):
		'''
		get waveform_upload_chache object
		Args:
			key (str) : name of the channel
		'''
		# cdef pair[string, int] awg_loc
		# awg_loc = self.channel_to_AWG_map[key.encode()]
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