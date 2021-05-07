import warnings
import numpy as np

from dataclasses import dataclass

from qcodes.instrument.parameter import Parameter

class virtual_gates_constructor(object):
    """
    Construtor to initialize virtual gate matrixes.
    """
    def __init__(self, pulse_lib_obj, name=None):
        """
        init object
        Args:
            pulse_lib_obj (pulse_lib) : add a pulse lib object to whom properties need to be added.
        """
        self.pulse_lib_obj = pulse_lib_obj
        self.pulse_lib_obj.virtual_channels.append(self)
        self.name = name
        self.real_gate_names = []
        self.virtual_gate_names = []
        self._virtual_gate_matrix = None
        self.valid_indices = None

    @property
    def virtual_gate_matrix(self):
        if self._virtual_gate_matrix is None:
            raise ValueError("Cannot fetch virutal gate matrix, please define (see docs).")

        self.virtual_gate_matrix_tmp = np.asarray(self._virtual_gate_matrix)
        self.virtual_gate_matrix_tmp = self.virtual_gate_matrix_tmp[self.valid_indices]
        self.virtual_gate_matrix_tmp = self.virtual_gate_matrix_tmp[:,self.valid_indices]
        return self.virtual_gate_matrix_tmp

    @property
    def virtual_gate_matrix_inv(self):
        return np.linalg.inv(self.virtual_gate_matrix)


    @property
    def size(self):
        """
        size of the virtual gate matrix
        """
        return self.virtual_gate_matrix.shape[0]

    def load_via_harware(self, virtual_gate_set):
        '''
        load in a virtual gate set by the hardware file.
        This has as advantage that when the matrix get updated for the dac, the same happens for the AWG channels.

        Args:
            virtual_gate_set (virtual_gate) : virtual_gate object
        '''
        # fetch gates that are also present on the AWG.
        idx_of_valid_gates = []
        for i in range(len(virtual_gate_set)):
            if virtual_gate_set.real_gate_names[i] in self.pulse_lib_obj.awg_channels:
                idx_of_valid_gates.append(i)

        if len(idx_of_valid_gates) == 0:
            warnings.warn("No valid gates found of the AWG for the virtual gate set {}. This virtual gate entry will be neglected.".format(virtual_gate_set.name))
            return

        self.valid_indices = np.array(idx_of_valid_gates, dtype=np.int)
        self._virtual_gate_matrix = virtual_gate_set.virtual_gate_matrix
        self.real_gate_names = list(np.asarray(virtual_gate_set.real_gate_names)[idx_of_valid_gates])
        self.virtual_gate_names =list( np.asarray(virtual_gate_set.virtual_gate_names)[idx_of_valid_gates])

    def load_via_hardware_new(self, virtual_gate_set):
        idx_of_valid_gates = []
        for i in range(len(virtual_gate_set)):
            if virtual_gate_set.gates[i] in self.pulse_lib_obj.awg_channels:
                idx_of_valid_gates.append(i)

        if len(idx_of_valid_gates) == 0:
            warnings.warn("No valid gates found of the AWG for the virtual gate set {}. This virtual gate entry will be neglected.".format(virtual_gate_set.name))
            return

        self.valid_indices = np.array(idx_of_valid_gates, dtype=np.int)
        self._virtual_gate_matrix = virtual_gate_set.matrix
        self.real_gate_names = list(np.asarray(virtual_gate_set.gates)[idx_of_valid_gates])
        self.virtual_gate_names =list( np.asarray(virtual_gate_set.v_gates)[idx_of_valid_gates])

    def add_real_gates(self, *args):
        """
        specify list of real gate names where to map on in the virtual gate matrix (from left to the right in the matrix)
        Args:
            *args (str) : naems of AWG channels to map on to.
        """
        for i in args:
            if i not in self.pulse_lib_obj.awg_channels:
                raise ValueError("{} non declared in the pulse_lib object. Make sure to specify the channel.".format(i))

        self.real_gate_names = args

    def add_virtual_gates(self, *args):
        """
        specify list of virtual gate names where to map on in the virtual gate matrix (from left to the right in the matrix)
        Args:
            *args (str) : names of virtual AWG channels to map on to.
        """
        for i in args:
            if i in self.pulse_lib_obj.awg_channels:
                raise ValueError("Name error, a virtual gate should have a different name that the one of a real one {}.".format(i))
        self.virtual_gate_names = args

    def add_virtual_gate_matrix(self, virtual_gate_matrix):
        """
        add the inverted virtual gate matrix.
        Args :
            virtual_gate_matrix (np.ndarray[ndim=2, type=double]) : numpy array representing the inverted virtual gate matrix.
        """
        n = virtual_gate_matrix.shape[0]

        if n != len(self.real_gate_names):
            raise ValueError("size virtual gate matrix ({}) not matching the given amount of real gates names({})".format(n, len(self.real_gate_names)))
        if n != len(self.virtual_gate_names):
            raise ValueError("size virtual gate matrix ({}) not matching the given amount of virutal gates names({})".format(n, len(self.virtual_gate_names)))

        self.valid_indices = np.arange(n, dtype=np.int)
        self._virtual_gate_matrix = virtual_gate_matrix.data


@dataclass
class marker_info:
    """
    structure to save relevant information about marker data.
    """
    Marker_channel: str

@dataclass
class virtual_pulse_channel_info:
    """
    info that is needed to link a real channel to a virtual channel
    """
    name: str
    multiplication_factor: float
    seg_container: any

    @property
    def segment(self):
        return getattr(self.seg_container, self.name)


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
        """
        get LO frequecy of the MW source
        """
        if isinstance(self._LO, float):
            return self._LO
        elif isinstance(self._LO, Parameter):
            return self._LO.get()
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

        self.__check_awg_channel_name(channel_name)

        IQ_comp = IQ_comp.upper()
        if IQ_comp not in ["I", "Q"]:
            raise ValueError("The compenent of the IQ signal is not specified properly (current given {}, expected \"I\" or \"Q\")".format(IQ_comp))

        if image not in ["+", "-"]:
            raise ValueError("The image of the IQ signal is not specified properly (current given {}, expected \"+\" or \"-\")".format(image))

        self.IQ_channel_map.append(IQ_channel_info(channel_name, IQ_comp, image))

    def add_marker(self, channel_name, pre_delay=0, post_delay=0):
        """
        Channel for in phase information of the IQ channel (postive image)
        Args:
            channel_name (str) : name of the channel in the AWG used to output
            pre_delay (float) : number of ns that the marker needs to be send before the IQ pulse.
            post_delay (float) : how long to keep the marker on after the IQ pulse is done.
        """
        if pre_delay or post_delay:
            raise Exception(f'delays must be set with pulse_lib.define_marker(name, setup_ns=pre_delay, hold_ns=post_delay)')
        self.__check_marker_channel_name(channel_name)
        self.markers.append(marker_info(channel_name))

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

    def __check_awg_channel_name(self, channel_name):
        """
        quickly checks that if the channel that is given has a valid name.
        Args:
            channel_name (str) : name of the physical channel to check
        """
        if channel_name not in self.pulse_lib_obj.awg_channels:
            raise ValueError("The given channel {} does not exist. Please specify first the physical channels on the AWG and then make the virtual ones.".format(channel_name))

    def __check_marker_channel_name(self, channel_name):
        """
        quickly checks that if the channel that is given has a valid name.
        Args:
            channel_name (str) : name of the physical channel to check
        """
        if channel_name not in self.pulse_lib_obj.marker_channels:
            raise ValueError("The given channel {} does not exist. Please specify first the physical channels on the AWG and then make the virtual ones.".format(channel_name))
