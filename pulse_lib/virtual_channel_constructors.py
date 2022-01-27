import numpy as np

from .configuration.iq_channels import IQ_channel, IQ_out_channel_info


class virtual_gates_constructor(object):
    """
    Constructor to create virtual gate matrixes.
    """
    def __init__(self, pulse_lib_obj, name=None):
        """
        init object
        Args:
            pulse_lib_obj (pulse_lib) : add a pulse lib object to whom properties need to be added.
            name (str): name of the matrix
        """
        self.pulse_lib_obj = pulse_lib_obj
        self.name = name
        self.real_gate_names = []
        self.virtual_gate_names = []
        # NOTE: _matrix_data is an external reference. The inverse cannot be cached.
        self._matrix = None

    def add_real_gates(self, *args):
        """
        specify list of real gate names where to map on in the
        virtual gate matrix (from left to the right in the matrix)
        Args:
            *args (str) : naems of AWG channels to map on to.
        """
        defined_channels = self.pulse_lib_obj.channels
        for i in args:
            if i not in defined_channels:
                raise ValueError(f"Channel {i} not defined in the pulse_lib object.")

        self.real_gate_names = args

    def add_virtual_gates(self, *args):
        """
        specify list of virtual gate names where to map on in the virtual gate matrix (from left to the right in the matrix)
        Args:
            *args (str) : names of virtual AWG channels to map on to.
        """
        self.virtual_gate_names = []
        defined_channels = self.pulse_lib_obj.channels
        for i in args:
            if i in defined_channels:
                raise ValueError(f"Cannot add virtual gate {i}. There is another gates with the same name.")
        self.virtual_gate_names = args

    def add_virtual_gate_matrix(self, virtual_gate_matrix):
        """
        add the inverted virtual gate matrix.
        Args :
            virtual_gate_matrix (np.ndarray[ndim=2, type=double]) : numpy array representing the (inverted) virtual gate matrix.
        """
        self._matrix = virtual_gate_matrix

        self._update_virtual_gate_matrix()

    def _update_virtual_gate_matrix(self):
        if self._matrix is None:
            return

        self.pulse_lib_obj.add_virtual_matrix(
                self.name,
                self.real_gate_names,
                self.virtual_gate_names,
                self._matrix,
                real2virtual=True,
                filter_undefined=True,
                keep_squared=True)


class IQ_channel_constructor(object):
    """
    Constructor that makes virtual IQ channels on the AWG instruments.
    Recommended to construct if you plan to use and MW control.
    """
    def __init__(self, pulse_lib_obj, name=None):
        """
        init object
        Args:
            pulse_lib_obj (pulse_lib) : add a pulse lib object to whom properties need to be added.
        """
        self.pulse_lib_obj = pulse_lib_obj
        if name is None:
            name = f'_IQ-{len(pulse_lib_obj.IQ_channels)}'
        self.IQ_channel:IQ_channel = pulse_lib_obj.define_IQ_channel(name)

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

        self.IQ_channel.IQ_out_channels.append(IQ_out_channel_info(channel_name, IQ_comp, image))

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
        self.IQ_channel.marker_channels.append(channel_name)

    def set_LO(self, LO):
        """
        Set's frequency of the microwave source --> the local oscilator.
        Args:
            LO (Parameter/float) :
        """
        self.IQ_channel.LO_parameter = LO

    def add_virtual_IQ_channel(self, virtual_channel_name,
                               LO_freq=None,
                               correction_phase=0.0,
                               correction_gain=(1.0,1.0)):
        """
        Make a virtual channel that hold IQ signals. Each virtual channel can hold their own phase information.
        It is recommended to make one IQ channel per qubit (assuming you are multiplexing for multiple qubits)
        Args:
            virtual_channel_name (str) : channel name (e.g. qubit_1)
            LO_freq (float) : frequency of the qubit when not driving and default for driving.
            correction_phase (float) : phase in rad added to Q component of IQ channel
            correction_gain (float) : correction of I and Q gain
        """
        self.pulse_lib_obj.define_qubit_channel(virtual_channel_name, self.IQ_channel.name,
                                                reference_frequency=LO_freq,
                                                correction_phase=correction_phase,
                                                correction_gain=correction_gain)


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
