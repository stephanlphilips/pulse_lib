from pulse_lib.segments.segments import segment_container
from pulse_lib.segments.data_handling_functions import find_common_dimension
from pulse_lib.keysight.uploader import upload_job

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
		self.id = None
		self.units = None
		self.setpoints = None
		self.names = None
		self.units = None

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

	@property
	def shape(self):
		return self._shape
	
	@property
	def ndim(self):
		return len(self.shape)

	def add_sequence(self, sequence):
		'''
		adds a sequence to this object. The Sequence needs to be defined like:
		Args:
			sequence (array) : array of arrays with in the latter one,
				[segmentobject, n_rep, prescaler] (n_rep and prescaler are by default one)
		'''
		# correct format if needed
		for i in range(len(sequence)):
			if isinstance(sequence[i], segment_container):
				self.sequence.append([sequence[i], 1, 1])
			elif isinstance(sequence[i], list):
				self.sequence.append(sequence[i])
			else:
				raise ValueError('The provided element in the sequence seems to be of the wrong data type. {} provided, but segment_container or list expected'.format(type(sequence[i])))
		
		# update dimensionality of all sequennce objects
		for i in self.sequence:
			self._shape = find_common_dimension(i[0].shape, self._shape)

		self._shape = tuple(self._shape)

		for i in self.sequence:
			# NOTE : referencing going on here no copy!
			# TODO : check if there is a need to revert the dimension back when done with the experiment.
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
			compenstate (bool) : compenstate yes or no (default if not set it yes)
		'''
		self.neutralize = compenstate

	def add_hvi(self, HVI):
		'''
		Add HVI code to the AWG.
		Args:
			HVI (path) : location of the HVI file to upload, before doing the experiment.
		'''
		self.HVI = HVI

	def upload(self, index):
		'''
		Sends the sequence with the provided index to the uploader module. Once he is done, the play function can do its work.
		Args:
			index (tuple) : index if wich you wannt to upload. This index should fit into the shape of the sequence being played.
		
		Remark that upload and play can run at the same time and it is best to
		start multiple uploads at once (during upload you can do playback, when the first one is finihsed)
		(note that this is only possible if you AWG supports upload while doing playback)
		'''
		
		upload_object = upload_job(self.sequence, index, self.id, self.neutralize, self.priority)
		upload_object.add_dsp_function(self.DSP)

		self.uploader.add_upload_job(upload_object)



	def play(self, index):
		'''
		Playback a certain index, assuming the index is provided.
		Args:
			index (tuple) : index if wich you wannt to upload. This index should fit into the shape of the sequence being played.

		Note that the playback will not start until you have uploaded the waveforms.
		'''
		upload_data = self.uploader.get_upload_info(self.id, index)

		
		self._free_memory(index)

	def _free_memory(self, index):
		'''
		function to free up memory in the AWG manually. By default the sequencer class will do garbarge collection for you (e.g. delete waveforms after playback)
		Args:
			index (tuple) : index if wich you wannt to upload. This index should fit into the shape of the sequence being played.
		'''
		pass


