from dataclasses import dataclass
# from qcodes.intruments.parameter import Parameter


@dataclass
class marker_info:
	"""
	structure to save relevant information about marker data.
	"""
	Marker_channel: str
	pre_dalay: float = 0.0
	post_delay: float = 0.0

@dataclass
class IQ_channel_info:
	"""
	structure to save relevant information about the rendering of the IQ channels.
	"""
	channel_name: str
	# I or Q component
	IQ_comp: str
	# make the negative of positive image of the singal (*-1)
	image: str

@dataclass 
class virtual_channel_info:
	"""
	structure to hold info of virtual_IQ_channels
	"""
	channel_name : str
	reference_frequency : float

class IQ_channel_constructor(object):
	"""
	Constructor that makes virtual IQ channels on the AWG intruments.
	Recommended to construct if you plan to use and MW control.
	"""
	def __init__(self, pulse_lib_obj):
		"""
		init object
		Args:
			pulse_lib_obj (pulse_lib) : add a pulse lib object to whom properties need to be added.
		"""
		self.pulse_lib_obj = pulse_lib_obj
		self.pulse_lib_obj.IQ_channels.append(self)
		self.virtual_channel_map = []
		self.IQ_channel_map = []
		self.markers = []
		self._LO = None
	
	@property
	def LO(self):
		if isinstance(self._LO, Parameter):
			return self._LO.get()
		elif isinstance(self._LO, float):
			return self._LO
		else:
			raise ValueError("No local oscilator defined in the IQ_channel_constructor. Please do so.")

	def add_IQ_chan(self, channel_name, IQ_comp, image = "+"):
		"""
		Channel for in phase information of the IQ channel (postive image)
		Args:
			channel_name (str) : name of the channel in the AWG used to output
			IQ_comp (str) : "I" or "Q" singal that needs to be generated
			image (str) : "+" or "-", specify only when differential inputs are needed.
		"""

		self.__check_channel_name(channel_name)
		
		IQ_comp = IQ_comp.upper()
		if IQ_comp not in ["I", "Q"]:
			raise ValueError("The compenent of the IQ signal is not specified properly (current given {}, expected \"I\" or \"Q\")".format(IQ))

		if image not in ["+", "Q"]:
			raise ValueError("The image of the IQ signal is not specified properly (current given {}, expected \"+\" or \"-\")".format(IQ))
		
		self.IQ_channel_map.append(IQ_channel_info(channel_name, IQ_comp, image))

	def add_marker(self, channel_name, pre_delay, post_delay):
		"""
		Channel for in phase information of the IQ channel (postive image)
		Args:
			channel_name (str) : name of the channel in the AWG used to output
			pre_delay (float) : number of ns that the marker needs to be send before the IQ pulse.
			post_delay (float) : how long to keep the marker on after the IQ pulse is done.
		"""
		self.__check_channel_name(channel_name)
		self.markers.append(marker_info(channel_name, pre_delay, post_delay))

	def set_LO(self, LO):
		"""
		Set's frequency of the microwave source --> the local oscilator.
		Args:
			LO (Parameter/float) : 
		"""
		self._LO = LO

	def add_virtual_IQ_channel(self, virtual_channel_name, LO_freq = None):
		"""
		Make a virtual channel that hold IQ signals. Each virtual channel can hold their own phase information.
		It is recommended to make one IQ channel per qubit (assuming you are multiplexing for multiple qubits)
		Args:
			virtual_channel_name (str) : channel name (e.g. qubit_1)
			LO_freq (str) : can be used to specify a resting frequency of the qubit.
							This is handy when your qubit frequency is diferent in the reseting state and while driving.
							DEV NOTE? should this be here? (-- it is a general property to construct IQ pulses in a correct way --)
		"""
		self.virtual_channel_map.append(virtual_channel_info(virtual_channel_name, LO_freq))

	def __check_channel_name(self, channel_name):
		"""
		quickly checks that if the channel that is given has a valid name. 
		Args:
			channel_name (str) : name of the physical channel to check
		"""
		if channel_name not in self.pulse_lib_obj.awg_channels:
			raise ValueError("The given channel {} does not exist. Please specify first the physical channels on the AWG and then make the virtual ones.".format(channel_name))
		