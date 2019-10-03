import pulse_lib.segments.segment_container
from pulse_lib.segments.utility.data_handling_functions import find_common_dimension, update_dimension
from pulse_lib.segments.utility.setpoint_mgr import setpoint_mgr
from pulse_lib.segments.utility.looping import loop_obj
from pulse_lib.keysight.uploader import upload_job,convert_prescaler_to_sample_rate
from pulse_lib.segments.data_classes.data_HVI_variables import marker_HVI_variable
from pulse_lib.segments.data_classes.data_generic import data_container
from si_prefix import si_format

import numpy as np
import uuid

class sequencer():
	"""
	Class to make sequences for segments.
	"""
	def __init__(self, upload_module, correction_limits):
		'''
		make a new sequence object.
		Args:
			upload_module (uploader) : class of the upload module. Used to submit jobs
			correction_limits (dict) : dict that contains the limits in voltage that can be used to correct the waveform amplitude at the end of a sequence.
		Returns:
			None
		'''
		# each segment had its own unique identifier.
		self.id = uuid.uuid4()
		
		self._units = None
		self._setpoints = None
		self._names = None

		self._shape = (1,)
		self.sequence = list()
		self.uploader = upload_module
		self.correction_limits = correction_limits
		
		# arguments of post processing the might be needed during rendering.
		self.DSP = None
		self.neutralize = True
		self.priority = -1
		
		# HVI if needed..
		self.HVI = None
		self.HVI_compile_function = None
		self.HVI_start_function = None
		self.HVI_kwargs = None

		self.n_rep = 1000
		self.prescaler = 0
		self._sample_rate = 1e9
		self._HVI_variables = None


	@property
	def shape(self):
		return self._shape
	
	@property
	def ndim(self):
		return len(self.shape)

	@property
	def setpoint_data(self):
		setpoint_data = setpoint_mgr()
		for seg_container in self.sequence:
			setpoint_data += seg_container[0].setpoint_data

		return setpoint_data

	@property
	def units(self):
		return self.setpoint_data.units
	
	@property
	def labels(self):
		return self.setpoint_data.labels

	@property
	def setpoints(self):
		return self.setpoint_data.setpoints
	
	@property
	def HVI_variables(self):
		"""
		object that contains variable that can be ported into HVI.
		"""
		return self._HVI_variables

	@property
	def sample_rate(self):
		return self._sample_rate

	@sample_rate.setter
	def sample_rate(self, rate):
		"""
		Rate at which to set the AWG. Note that not all rates are supported and a rate as close to the one you enter will be put.
		
		Args:
			rate (float) : target sample rate for the AWG (unit : S/s).
		"""
		if rate > 200e6:
			prescaler = 0
		elif rate > 50e6:
			prescaler = 1
		else:
			prescaler = 1e9/(5*rate*2)

		self.prescaler = int(prescaler)
		self._sample_rate = convert_prescaler_to_sample_rate(prescaler)

		print("Info : effective sampling rate is set to {}S/s".format(si_format(self._sample_rate, precision=1)))


	def add_sequence(self, sequence):
		'''
		adds a sequence to this object. The Sequence needs to be defined like:
		Args:
			sequence (array) : array of arrays with in the latter one,
				[segmentobject, n_rep, prescaler] (n_rep and prescaler are by default one)
		'''
		# correct format if needed
		for i in range(len(sequence)):
			if isinstance(sequence[i], pulse_lib.segments.segment_container.segment_container):
				self.sequence.append([sequence[i], 1, 1])
			elif isinstance(sequence[i], list):
				self.sequence.append(sequence[i])
			else:
				raise ValueError('The provided element in the sequence seems to be of the wrong data type. {} provided, but segment_container or list expected'.format(type(sequence[i])))
		
		# update dimensionality of all sequence objects
		for i in self.sequence:
			i[0].enter_rendering_mode()
			self._shape = find_common_dimension(i[0].shape, self._shape)

		self._shape = tuple(self._shape)
		self._HVI_variables = data_container(marker_HVI_variable())
		self._HVI_variables = update_dimension(self._HVI_variables, self.shape)

		# enforce master clock for the current segments (affects the IQ channels (translated into a phase shift) and and the marker channels (time shifts))
		t_start = 0
		t_tot = np.zeros(self.shape)

		for i in self.sequence:
			segment_container = i[0]
			segment_container.extend_dim(self._shape, ref=True)

			lp_time = loop_obj()
			lp_time.add_data(t_tot, axis=list(range(self.ndim -1,-1,-1)))
			segment_container.add_master_clock(lp_time)
			self._HVI_variables += segment_container._software_markers.pulse_data_all

			t_tot += segment_container.total_time

	def add_dsp(self, dps_corr):
		'''
		Add a class to be used for dsp corrections (note only IIR and FIR allowed for performace reasons)
		Args: 
			dps_corr (dps_corr_class) : object that can be used to perform the DSP correction
		'''
		self.DSP = dps_corr

	def voltage_compenstation(self, compenstate):
		'''
		add a voltage compenstation at the end of the sequence
		Args:
			compenstate (bool) : compenstate yes or no (default is True)
		'''
		self.neutralize = compenstate

	def add_HVI(self, HVI_ID ,HVI_to_load, compile_function, start_function, **kwargs):
		'''
		Add HVI code to the AWG.
		Args:
			HVI_ID (str) : string that gives an ID to the HVI that is currently loaded.
			HVI_to_load (function) : function that returns a HVI file.
			compile_function (function) : function that compiles the HVI code. Default arguments that will be provided are (HVI, npt, n_rep) = (HVI object, number of points of the sequence, number of repetitions wanted)
			start_function (function) : function to be executed to start the HVI (this can also be None)
			kwargs : keyword arguments for the HVI script (see usage in the examples (e.g. when you want to provide your digitzer card))
		'''
		if self.uploader.current_HVI_ID != HVI_ID :
			self.HVI = HVI_to_load(self.uploader.AWGs, self.uploader.channel_map, **kwargs)
			self.uploader.current_HVI_ID = HVI_ID
			self.uploader.current_HVI = self.HVI
		else: 
			self.HVI = self.uploader.current_HVI

		self.HVI_compile_function = compile_function
		self.HVI_start_function = start_function
		self.HVI_kwargs = kwargs

	def upload(self, index):
		'''
		Sends the sequence with the provided index to the uploader module. Once he is done, the play function can do its work.
		Args:
			index (tuple) : index if wich you wannt to upload. This index should fit into the shape of the sequence being played.
		
		Remark that upload and play can run at the same time and it is best to
		start multiple uploads at once (during upload you can do playback, when the first one is finihsed)
		(note that this is only possible if you AWG supports upload while doing playback)
		'''
		
		upload_object = upload_job(self.sequence, index, self.id, self.n_rep ,self.prescaler, self.neutralize, self.priority)
		if self.DSP is not None:
			upload_object.add_dsp_function(self.DSP)
		if self.HVI is not None:
			upload_object.add_HVI(self.HVI, self.HVI_compile_function, self.HVI_start_function, **{**self.HVI_kwargs, **self._HVI_variables.item(tuple(index)).HVI_markers})

		self.uploader.add_upload_job(upload_object)


	def play(self, index, release= True):
		'''
		Playback a certain index, assuming the index is provided.
		Args:
			index (tuple) : index if wich you wannt to upload. This index should fit into the shape of the sequence being played.
			release (bool) : release memory on the AWG after the element has been played.

		Note that the playback will not start until you have uploaded the waveforms.
		'''
		self.uploader.play(self.id, index, release)
		
	def release_memory(self, index):
		'''
		function to free up memory in the AWG manually. By default the sequencer class will do garbarge collection for you (e.g. delete waveforms after playback)
		Args:
			index (tuple) : index if wich you wannt to upload. This index should fit into the shape of the sequence being played.
		'''
		self.uploader.release_memory(self.id, index)



if __name__ == '__main__':
	from pulse_lib.segments.segment_container import segment_container
	import pulse_lib.segments.utility.looping as lp

	a = segment_container(["a", "b"])
	b = segment_container(["a", "b"])

	b.a.add_block(0,lp.linspace(30,100,10),100)
	b.a.reset_time()
	a.add_HVI_marker("marker_name", 20)
	b.add_HVI_marker("marker_name2", 50)

	b.add_HVI_variable("my_vatr", 800)
	a.a.add_block(20,lp.linspace(50,100,10, axis = 1, name = "time", unit = "ns"),100)

	b.slice_time(0,lp.linspace(80,100,10, name = "time", unit = "ns", axis= 2))
	
	my_seq = [a,b]

	seq = sequencer(None, dict())
	seq.add_sequence(my_seq)
	print(seq.HVI_variables.flat[0].HVI_markers)
	# print(seq.labels)
	# print(seq.units)
	# print(seq.setpoints)
	seq.upload([0])