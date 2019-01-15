from libcpp cimport bool
from libc.stdlib cimport malloc, free

cdef extern from "SD_WAVE.h":
	cdef cppclass SD_Wave:
		SD_Wave(const char *waveformFile, char *name = 0);
		SD_Wave(int waveformType, int waveformPoints, double *waveformDataA, double *waveformDataB = 0);
		SD_Wave(int waveformType, int waveformPoints, int *waveformDataA, int *waveformDataB = 0);
		SD_Wave(const SD_Wave *waveform);

		short *getPointVector() const;
		int getPoints() const;
		int getStatus() const;
		int getType() const;

cdef extern from "SD_Module.h":
	cdef cppclass SD_Module:
		SD_Module(int)
		int moduleCount()
		int getProductName(int , int , char *)
		int getProductName(int , char *)
		int getSerialNumber(int , int , char *)
		int getSerialNumber(int , char *)
		int getType(int , int )
		int getType(int )
		int getChassis(int )
		int getSlot(int )

		int open(const char *partNumber, const char *serialNumber);
		int open(const char *partNumber, int nChassis, int nSlot);
		int open(const char* productName, const char* serialNumber, int compatibility);
		int open(const char* productName, int chassis, int slot, int compatibility);
		bool isOpened() const;
		int close();

		int runSelfTest();
		int getStatus() const;
		char *getSerialNumber(char *serialNumber) const;
		char *getProductName(char *productName) const;
		double getFirmwareVersion() const;
		double getHardwareVersion() const;
		int getChassis() const;
		int getSlot() const;
		const char *moduleName() const;

		# # //FPGA
		# int FPGAreadPCport(int port, int *buffer, int nDW, int address, SD_AddressingMode::SD_AddressingMode addressMode = SD_AddressingMode::AUTOINCREMENT, SD_AccessMode::SD_AccessMode accessMode = SD_AccessMode::DMA);
		# int FPGAwritePCport(int port, int *buffer, int nDW, int address, SD_AddressingMode::SD_AddressingMode addressMode = SD_AddressingMode::AUTOINCREMENT, SD_AccessMode::SD_AccessMode accessMode = SD_AccessMode::DMA);
		# int FPGAload(const char *fileName);
		# int FPGAreset(SD_ResetMode::SD_ResetMode mode = SD_ResetMode::PULSE);

		# //HVI Variables
		int readRegister(int varNumber, int &errorOut) const;
		int readRegister(const char *varName, int &errorOut) const;
		double readRegister(int varNumber, const char *unit, int &errorOut) const;
		double readRegister(const char *varName, const char *unit, int &errorOut) const;
		int writeRegister(int varNumber, int varValue);
		int writeRegister(const char *varName, int varValue);
		int writeRegister(int varNumber, double value, const char *unit);
		int writeRegister(const char *varName, double value, const char *unit);

		# //PXItrigger
		int PXItriggerWrite(int nPXItrigger, int value);
		int PXItriggerRead(int nPXItrigger) const;

		# //DAQ
		int DAQconfig(int nDAQ, int nDAQpointsPerCycle, int nCycles, int prescaler, int triggerMode);
		int DAQbufferPoolRelease(int nDAQ);
		int DAQcounterRead(int nDAQ) const;
		int DAQtrigger(int nDAQ);
		int DAQstart(int nDAQ);
		int DAQpause(int nDAQ);
		int DAQresume(int nDAQ);
		int DAQflush(int nDAQ);
		int DAQstop(int nDAQ);
		int DAQtriggerMultiple(int DAQmask);
		int DAQstartMultiple(int DAQmask);
		int DAQpauseMultiple(int DAQmask);
		int DAQresumeMultiple(int DAQmask);
		int DAQflushMultiple(int DAQmask);
		int DAQstopMultiple(int DAQmask);

		# //Extenal Trigger
		int translateTriggerPXItoExternalTriggerLine(int trigger) const;
		int translateTriggerIOtoExternalTriggerLine(int trigger) const;
		int WGtriggerExternalConfig(int nAWG, int externalSource, int triggerBehavior, bool sync = true);
		int DAQtriggerExternalConfig(int nDAQ, int externalSource, int triggerBehavior, bool sync = false);

		# //AWG
		int waveformGetAddress(int waveformNumber);
		int waveformGetMemorySize(int waveformNumber);
		int waveformMemoryGetWriteAddress();
		int waveformMemorySetWriteAddress(int writeAddress);

		int waveformReLoad(int waveformType, int waveformPoints, short *waveformDataRaw, int waveformNumber, int paddingMode = 0);
		int waveformReLoad(SD_Wave *waveformObject, int waveformNumber, int paddingMode = 0);

		int waveformLoad(int waveformType, int waveformPoints, short *waveformDataRaw, int waveformNumber, int paddingMode = 0);
		int waveformLoad(SD_Wave *waveformObject, int waveformNumber, int paddingMode = 0);
		int waveformFlush();

		# //HVI management
		int openHVI(const char *fileHVI);
		int compileHVI();
		int compilationErrorMessageHVI(int errorIndex, char *message, int maxSize);
		int loadHVI();

		# //HVI Control
		int startHVI();
		int pauseHVI();
		int resumeHVI();
		int stopHVI();
		int resetHVI();

		# // P2P
		int DAQp2pStop(int nChannel);
		unsigned long long pipeSinkAddr(int nPipeSink) const;
		int DAQp2pConfig(int nChannel, int dataSize, int timeOut, unsigned long long pipeSink) const;



ctypedef SD_Module* SD_Module_ptr
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.pair cimport pair
from libcpp.map cimport map as mapcpp

import numpy as np
cimport numpy as np

cdef struct s_waveform_info:
	pair[double, double] min_max_voltage
	double integral

ctypedef s_waveform_info waveform_info

cdef class s_waveform_raw_upload_data:
	cdef vector[double[:]] *wvf_data
	# np added here already as you cannot fetch it without the gil 
	cdef vector[int] *wvf_npt
	cdef pair[double, double] *min_max_voltage
	cdef vector[double[:]] *DSP_param
	cdef short *upload_data
	cdef int npt



ctypedef s_waveform_raw_upload_data waveform_raw_upload_data


cdef class keysight_upload_module():
	"""
	This is a speraturate module for the upload in keysight units. This module also does some last post processing on the waveform (e.g. DSP/convert to short/extend them, so they fit into the memory ...)
	The upload in writtin in C, so can be fully run without gil in a multhithreaded way. 
	This module is in now way a replacement for the python module and has as main function to get the waveform into the memory
	"""
	cdef mapcpp[string, SD_Module_ptr] AWG_modules
	cdef int error_handle

	def add_awg_module(self, str name, int chassis, int slot):
		cdef SD_Module* awg_module
		cdef char* module_name

		awg_module = new SD_Module(0)
		
		module_name = ""
		self.error_handle = awg_module.getProductName(chassis, slot, module_name)
		self.__check_error()

		self.error_handle = awg_module.open(module_name, chassis, slot, 1)
		self.__check_error()

		self.AWG_modules[name.encode()] = awg_module

	def add_upload_data(self, waveform_cache_container waveform_cache):
		cdef mapcpp[string, mapcpp[int, waveform_raw_upload_data]] *AWG_raw_upload_data
		AWG_raw_upload_data = &waveform_cache.AWG_raw_upload_data
		

		# make cache's
		cdef int n_points = waveform_cache.npt

		cdef np.ndarray[dtype=short, ndim=1] waveform_data_raw = np.empty((n_points,), dtype=np.short)
		cdef short[:] waveform_data_raw_view = waveform_data_raw


		# from here all in in c!
		for i in range(AWG_raw_upload_data.size()):
			# Already assign memory for the final data
			pass

	cdef void rescale_concateneate_and_convert_to_16_bit_number(self, waveform_raw_upload_data* upload_data) nogil:
		'''
		low level function the voltages to 0/1 range and making a conversion to 16 bits.
		All voltages are also concatenated
		'''


		upload_data.upload_data = <short *> malloc(upload_data.npt * sizeof(short))

		cdef int i, idx_view, wvf_id  = 0

		cdef double[:] wvf_ptr
		cdef double v_offset = (upload_data.min_max_voltage.second + upload_data.min_max_voltage.first)/2
		cdef double v_pp = upload_data.min_max_voltage.second - upload_data.min_max_voltage.first

		for wvf_id in range(upload_data.wvf_npt.size()):
			# wvf_ptr = upload_data.wvf_data.at(wvf_id)
		# 	for idx_view in range(upload_data.wvf_npt.at(wvf_id)):
		# 		upload_data.upload_data[i] = <short> (( wvf_ptr[idx_view] - v_offset)/v_pp)
		# 		i+= 1
			pass
			
	# cdef void getzvpp_voff(self, waveform_raw_upload_data* upload_data, double *vpp, double *voff):
	# 	'''
	# 	Calculate the total voltage offset and peak to peak voltage to be uploaded.
	# 	'''
	# 	cdef double Vmin = upload_data.wvf_info[0].min_max_voltage.first
	# 	cdef double Vmin = upload_data.wvf_info[0].min_max_voltage.first



	cdef __check_error(self):
		if self.error_handle != 0:
			raise ValueError(self.error_handle)


cdef class waveform_upload_chache():
	"""object that holds some cache for uploads and does some basic calculations"""
	cdef vector[waveform_info] wvf_info
	cdef vector[double[:]] wvf_data
	cdef vector[double[:]] wvf_npt
	cdef pair[double, double] min_max_voltage
	cdef pair[double, double] compenstation_limit
	cdef int _npt
	def __init__(self, compenstation_limit):
		self.compenstation_limit = compenstation_limit
		self._npt = 0

	def add_data(self, wvf, v_min_max, integral):
		'''
		wvf (np.ndarray[ndim = 1, dtype=double]) : waveform
		v_min_max (tuple) : maximum/minimum voltage of the current segment
		integral (double) : integrated value of the waveform
		'''
		cdef waveform_info data_info
		cdef double[:] wvf_ptr = wvf
		
		if self._npt == 0:
			self.min_max_voltage = v_min_max
		else:
			if self.min_max_voltage.first > v_min_max[0]:
				self.min_max_voltage.first = v_min_max[0]
			if self.min_max_voltage.second < v_min_max[1]:
				self.min_max_voltage.second = v_min_max[1]

		self._npt += wvf_ptr.size

		data_info.min_max_voltage = v_min_max
		data_info.integral = integral

		self.wvf_info.push_back(data_info)
		self.wvf_data.push_back(wvf_ptr)
		self.wvf_npt.push_back(wvf_ptr.size)

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