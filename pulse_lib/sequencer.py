import pulse_lib.segments.segment_container
from pulse_lib.segments.utility.data_handling_functions import find_common_dimension
from pulse_lib.segments.utility.setpoint_mgr import setpoint_mgr
from pulse_lib.keysight.uploader import upload_job,convert_prescaler_to_sample_rate
from si_prefix import si_format

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
	def sample_rate(self):
		return self._sample_rate

	@sample_rate.setter
	def sample_rate(self, rate):
		"""
		Rate at which to set the AWG. Note that not all rates are supported and a rate as close to the one you enter will be put.
		
		Args:
			rate (float) : target sample rate for the AWG.
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
		
		# update dimensionality of all sequennce objects
		for i in self.sequence:
			i[0].enter_rendering_mode()
			self._shape = find_common_dimension(i[0].shape, self._shape)

		self._shape = tuple(self._shape)

		for i in self.sequence:
			i[0].extend_dim(self._shape, ref=True)

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

	def add_HVI(self, HVI_to_load, compile_function, start_function, **kwargs):
		'''
		Add HVI code to the AWG.
		Args:
			HVI_to_load (function) : function that returns a HVI file.
			compile_function (function) : function that compiles the HVI code. Default arguments that will be provided are (HVI, npt, n_rep) = (HVI object, number of points of the sequence, number of repetitions wanted)
			start_function (function) : function to be executed to start the HVI (this can also be None)
			kwargs : keyword arguments for the HVI script (see usage in the examples (e.g. when you want to provide your digitzer card))
		'''
		self.HVI = HVI_to_load(self.uploader.AWGs, self.uploader.channel_map, **kwargs)
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
			upload_object.add_HVI(self.HVI, self.HVI_compile_function, self.HVI_start_function, **self.HVI_kwargs)

		self.uploader.add_upload_job(upload_object)



	def play(self, index):
		'''
		Playback a certain index, assuming the index is provided.
		Args:
			index (tuple) : index if wich you wannt to upload. This index should fit into the shape of the sequence being played.

		Note that the playback will not start until you have uploaded the waveforms.
		'''
		self.uploader.play(self.id, index)
		
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

	a.a.add_block(20,lp.linspace(50,100,10, axis = 1, name = "time", unit = "ns"),100)

	b.slice_time(0,lp.linspace(80,100,10, name = "time", unit = "ns", axis= 2))
	
	my_seq = [a,b]

	seq = sequencer(None, dict())
	seq.add_sequence(my_seq)

	print(seq.labels)
	print(seq.units)
	print(seq.setpoints)