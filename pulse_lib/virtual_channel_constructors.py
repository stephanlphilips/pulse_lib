import warnings
import numpy as np

from .configuration.iq_channels import IQ_channel, IQ_out_channel_info

def add_detuning_channels(pulse, gate1, gate2, detuning, average):
    detuning_gate_set = virtual_gates_constructor(pulse, matrix_virtual2real=True)
    detuning_gate_set.add_real_gates(gate1, gate2)
    detuning_gate_set.add_virtual_gates(detuning, average)
    matrix = np.array([[+0.5, +1.0], [-0.5, -1.0]])
    detuning_gate_set.add_virtual_gate_matrix(matrix)


class virtual_gates_constructor(object):
    """
    Constructor to create virtual gate matrixes.
    """
    def __init__(self, pulse_lib_obj, name=None, matrix_virtual2real=False, square=True):
        """
        init object
        Args:
            pulse_lib_obj (pulse_lib) : add a pulse lib object to whom properties need to be added.
            name (str): name of the matrix
            matrix_virtual2real (bool): If True v_real = M @ v_virtual, else v_real = M^-1 @ v_virtual
            square (bool): matrix is square and should be kept square when a real gate is missing.
        """
        self.pulse_lib_obj = pulse_lib_obj
        self.pulse_lib_obj.virtual_channels.append(self)
        self.name = name
        self._square = square
        self.real_gate_names = []
        self.virtual_gate_names = []
        self.layer = 0
        self.valid_indices = None
        # NOTE: _matrix_data is an external reference. The inverse cannot be cached.
        self._matrix_data = None
        self._matrix_virtual2real = matrix_virtual2real
        # Set methods to get inverted or not inverted matrix.
        if matrix_virtual2real:
            self._virtual2real_matrix = self._matrix
            self._real2virtual_matrix = self._inv_matrix
        else:
            self._virtual2real_matrix = self._inv_matrix
            self._real2virtual_matrix = self._matrix


    @property
    def virtual2real_matrix(self):
        return self._virtual2real_matrix()

    @property
    def real2virtual_matrix(self):
        return self._real2virtual_matrix()

    # backwards compatibility method
    @property
    def virtual_gate_matrix(self):
        return self._real2virtual_matrix()

    # backwards compatibility method
    @property
    def virtual_gate_matrix_inv(self):
        return self._virtual2real_matrix()

    def _inv_matrix(self):
        return np.linalg.inv(self._matrix())

    def _matrix(self):
        if self._matrix_data is None:
            raise ValueError("Cannot fetch virtual gate matrix, please define (see docs).")

        matrix_tmp = np.asarray(self._matrix_data)

        if self._matrix_virtual2real:
            matrix_tmp = matrix_tmp[:,self.valid_indices]
            if self._square:
                matrix_tmp = matrix_tmp[self.valid_indices]
        else:
            matrix_tmp = matrix_tmp[self.valid_indices]
            if self._square:
                matrix_tmp = matrix_tmp[:,self.valid_indices]
        return matrix_tmp


    def load_via_harware(self, virtual_gate_set):
        '''
        load in a virtual gate set by the hardware file.
        This has as advantage that when the matrix get updated for the dac, the same happens for the AWG channels.

        Args:
            virtual_gate_set (virtual_gate) : virtual_gate object
        '''
        self._load(virtual_gate_set.name,
                   virtual_gate_set.real_gate_names,
                   virtual_gate_set.virtual_gate_names,
                   virtual_gate_set.virtual_gate_matrix)

    def load_via_hardware_new(self, virtual_gate_set):
        self._load(virtual_gate_set.name,
                   virtual_gate_set.gates,
                   virtual_gate_set.v_gates,
                   virtual_gate_set.matrix)

    def _load(self, name, real_gates, virtual_gates, matrix):
        # NOTE: loading must be done in proper order. Combination gates must be loaded after virtual gates

        self.virtual_gate_names = []
        defined_channels = self.pulse_lib_obj.channels

        # select gates that are also defined in pulselib
        idx_of_valid_gates = []
        not_defined_gates = []
        for i,name in enumerate(real_gates):
            if name in defined_channels:
                idx_of_valid_gates.append(i)
            else:
                not_defined_gates.append(name)

        if len(idx_of_valid_gates) == 0:
            warnings.warn(f"No valid gates found of the AWG for the virtual gate set {name}."
                          "This virtual gate entry will be neglected.")
            return

        if len(not_defined_gates):
            warnings.warn(f"Gates {not_defined_gates} of virtual gate set {name} "
                          "are not defined in pulselib.")

        self.valid_indices = np.array(idx_of_valid_gates, dtype=np.int)
        self._matrix_data = matrix
        self.real_gate_names = [real_gates[i] for i in idx_of_valid_gates]
        if self._square:
            self.virtual_gate_names = [virtual_gates[i] for i in idx_of_valid_gates]
        else:
            self.virtual_gate_names = virtual_gates

        self.layer = 1 + max(self.pulse_lib_obj.get_channel_layer(name)
                             for name in self.real_gate_names)

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
        self.layer = 1 + max(self.pulse_lib_obj.get_channel_layer(name)
                             for name in self.real_gate_names)

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
        shape = virtual_gate_matrix.shape
        n_real, n_virtual = (shape[0],shape[1]) if self._matrix_virtual2real else (shape[1],shape[0])

        if n_real != len(self.real_gate_names):
            raise ValueError(f"size virtual gate matrix ({n_real}) doesn't match "
                             f"the number of real gates names({len(self.real_gate_names)})")
        if n_virtual != len(self.virtual_gate_names):
            raise ValueError("size virtual gate matrix ({n_virtual}) doesn't match "
                             f"the number of virtual gates names({len(self.virtual_gate_names)})")

        self.valid_indices = np.arange(n_real, dtype=np.int)
        self._matrix_data = virtual_gate_matrix.data



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

    def add_virtual_IQ_channel(self, virtual_channel_name, LO_freq = None):
        """
        Make a virtual channel that hold IQ signals. Each virtual channel can hold their own phase information.
        It is recommended to make one IQ channel per qubit (assuming you are multiplexing for multiple qubits)
        Args:
            virtual_channel_name (str) : channel name (e.g. qubit_1)
            LO_freq (float) : frequency of the qubit when not driving and default for driving.
        """
        self.pulse_lib_obj.define_qubit_channel(virtual_channel_name, self.IQ_channel.name, reference_frequency=LO_freq)

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
