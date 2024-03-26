import logging
import numpy as np
import re
from typing import Dict, Union

from qcodes import Parameter

from pulse_lib.segments.segment_container import segment_container
from pulse_lib.sequencer import sequencer
from pulse_lib.configuration.physical_channels import (
        awg_channel, marker_channel, digitizer_channel, resonator_rf_source)
from pulse_lib.configuration.iq_channels import IQ_channel, QubitChannel
from pulse_lib.configuration.devices import awg_slave
from pulse_lib.configuration.rf_parameters import RfParameters
from pulse_lib.virtual_matrix.virtual_gate_matrices import VirtualGateMatrices

logger = logging.getLogger(__name__)


class pulselib:
    '''
    Global class that is an organisational element in making the pulses.
    The idea is that you first make individual segments,
    you can than later combine them into a sequence, this sequence will be uploaded
    '''

    def __init__(self, backend="Keysight"):
        self.awg_devices = dict()
        self.digitizers = dict()
        self.awg_channels = dict()
        self.marker_channels = dict()
        self.digitizer_channels = dict()
        self.rf_params = dict()
        self._virtual_matrices = VirtualGateMatrices()
        self.qubit_channels = dict()
        self.IQ_channels = dict()

        # Tektronix features
        self.digitizer_markers = dict()
        self.awg_sync = dict()

        # pulselib can configure digitizer if desired.
        self._configure_digitizer = False
        if backend == 'Qblox':
            # pulselib must configure digitizer for qblox
            self._configure_digitizer = True

        self._backend = backend
        self._default_hw_schedule_creator = None

        if np.__version__ < '1.20':
            raise Exception(f'Pulselib requires numpy 1.20+. Found version {np.__version__}')

    @property
    def channels(self):
        channels = []
        channels += self.awg_channels.keys()
        # Exclude marker_channels from channel list. channels property is used for 'sweepable' channels.
        # channels += self.marker_channels.keys()
        channels += self._virtual_matrices.virtual_gate_names
        return channels

    @property
    def configure_digitizer(self):
        return self._configure_digitizer

    @configure_digitizer.setter
    def configure_digitizer(self, enable):
        self._configure_digitizer = enable

    def add_awg(self, awg):
        '''
        add a awg to the library
        Args:
            awg (object) : qcodes AWG instrument
        '''
        self.awg_devices[awg.name] = awg

    def add_awgs(self, name, awg):
        '''
        add a awg to the library
        Args:
            name (str) : name you want to give to a peculiar AWG
            awg (object) : qcodes object of the concerning AWG
        '''
        if name != awg.name:
            raise Exception(f'name mismatch {name} != {awg.name}. Use awg.name or pulselib.add_awg(awg)')
        self.awg_devices[name] = awg

    def add_digitizer(self, digitizer):
        '''
        add a digitizer to the library
        Args:
            digitizer (object) : qcodes digitizer instrument
        '''
        self.digitizers[digitizer.name] = digitizer

    def add_digitizer_marker(self, digitizer_name, marker_name):
        '''
        Assign a marker as digitizer trigger
        Args:
            digitizer_name: name of the digitizer
            marker_name: name of the marker channel
        '''
        self.digitizer_markers[digitizer_name] = marker_name

    def add_awg_sync(self, awg_name, marker_name, sync_latency=None):
        '''
        Add synchronization for a slave AWG with a marker.
        Currently only used for Tektronix AWGs.

        Args:
            awg_name: name of the awg
            marker_name: name of the marker channel
            sync_latency (Optional[float]): latency in AWG triggering. If None it will be calculated from sample rate.

        Note:
            sync_latency of Tektronix AWG is sample frequency dependent.
        '''
        self.awg_sync[awg_name] = awg_slave(awg_name, marker_name, sync_latency)

    def define_channel(self, channel_name, AWG_name, channel_number, amplitude=None):
        '''
        define the channels and their location
        Args:
            channel_name (str) : name of a given channel on the AWG.
            AWG_name (str) : name of the instrument (as given in add_awgs())
            channel_number (int) : channel number on the AWG
            amplitude (float): (maximum) amplitude in mV. Uses instrument default when not set.

        Notes:
            For Keysight AWG the amplitude should only be set to enforce a maximum output level.
            The amplitude is applied digitally and setting it does not improve resolution of noise level.
            For Tektronix AWG the amplitude applies to the analogue output range.
            For Qblox setting the amplitude has no effect.
        '''
        self._check_uniqueness_of_channel_name(channel_name)
        self.awg_channels[channel_name] = awg_channel(channel_name, AWG_name, channel_number, amplitude)

    def define_marker(self, marker_name, AWG_name, channel_number, setup_ns=0, hold_ns=0,
                      amplitude=1000, invert=False):
        '''
        define the channels and their location
        Args:
            marker_name (str) : name of a given channel on the AWG.
            AWG_name (str) : name of the instrument (as given in add_awgs())
            channel_number (int or Tuple(int, int)) : channel number on the AWG
            setup_ns (float): setup time for the device using the marker. marker raises `setup_ns` earlier.
            hold_ns (float):
                hold time for the device using the marker to ensure proper output. marker falls `hold_ns` later.
            amplitude (float): amplitude in mV (only applies when instrument allows control)
            invert (bool): invert the ouput, i.e. high voltage when marker not set and low when marker is active.

        Notes:
            Keysight channel number 0 is trigger out, 1..4: use AWG analogue channel
            Tektronix: tuple, e.g. (1,2), defines marker output, integer uses analogue output channel
        '''
        self._check_uniqueness_of_channel_name(marker_name)
        self.marker_channels[marker_name] = marker_channel(marker_name, AWG_name, channel_number,
                                                           setup_ns, hold_ns, amplitude, invert)

    def define_digitizer_channel(self, name, digitizer_name, channel_number, iq_out=False,
                                 hw_input_channel=None):
        ''' Defines a digitizer channel.
        Args:
            channel_name (str): name of the channel.
            digitizer_name (str): name of digitizer
            channel_number (int): channel number
            iq_out (bool): if True output I+Q data, else output I data only.
            hw_input_channel (Optional[int]): channel number of input on hardware. (Keysight only)
        '''
        self._define_dig_channel(name, digitizer_name, [channel_number], iq_out=iq_out)
        if hw_input_channel is not None:
            self.set_digitizer_hw_input_channel(name, hw_input_channel)

    def define_digitizer_channel_iq(self, name, digitizer_name, channel_numbers, phase=0.0, iq_out=False):
        ''' Defines a digitizer I/Q input pair.
        Args:
            channel_name (str): name of the channel.
            digitizer_name (str): name of digitizer
            channel_numbers (List[int]): channel numbers: [I-channel, Q-channel]
            phase (float): phase shift in rad.
            iq_out (bool): if True output I+Q data, else output I data only.
        '''
        self._define_dig_channel(name, digitizer_name, channel_numbers,
                                 phase=phase,
                                 iq_out=iq_out, iq_input=True)

    def _define_dig_channel(self, name, digitizer_name, channel_numbers,
                            phase=0.0, iq_out=False, iq_input=False):
        self._check_uniqueness_of_channel_name(name)
        self.digitizer_channels[name] = digitizer_channel(name, digitizer_name, channel_numbers,
                                                          iq_out=iq_out, phase=phase,
                                                          iq_input=iq_input)
        self.rf_params[name] = RfParameters(self.digitizer_channels[name])

    def set_digitizer_phase(self, channel_name, phase):
        '''
        Sets phase of digitizer channel.
        Args:
            channel_name (str): name of the channel.
            phase (float): phase shift in rad.
        Note:  The phase applies only when the digitizer does the IQ demodulation
        '''
        self.digitizer_channels[channel_name].phase = phase

    def set_digitizer_frequency(self, channel_name, frequency):
        '''
        Sets frequency of digitizer channel for IQ demodulation.
        Args:
            channel_name (str): name of the channel.
            frequency (float): frequency in Hz.
        Note:  The phase applies only when the digitizer does the IQ demodulation
        '''
        self.digitizer_channels[channel_name].frequency = frequency

    def set_digitizer_hw_input_channel(self, channel_name, hw_channel_number):
        '''
        Sets hardware input channel for Keysight digitizer using FPGA demodulation.
        Args:
            channel_name (str): name of the channel.
            hw_channel_number (int): channel number of input on hardware.
        '''
        self.digitizer_channels[channel_name].hw_input_channel = hw_channel_number

    def set_digitizer_rf_source(self, channel_name, output,
                                mode='pulsed',
                                amplitude=0,
                                attenuation=1.0,
                                startup_time_ns=0,
                                prolongation_ns=0,
                                source_delay_ns=0,
                                ):
        '''
        Adds a resonator RF source to the digitizer channel.
        The resonator will be driven with the frequency specified for the digitizer
        channel and dependent on the mode can be enabled synchronous with acquisitions.

        The rf source can also be refer to a marker channel to enable an external modulator.

        Args:
            channel_name (str): name of the digitizer channel.
            output one of the following:
                (str) name of awg_channel
                (Tuple[str, int]) name of module and channel number
                (Tuple[str, List[int]]) name of module and channel numbers
            mode (str):
                'continuous' enables output from start of sequence till after last acquisition.
                'pulsed' enables output `startup_time_ns` before each acquisition till end of the acquisition.
                'shaped' generates a pulse equal to the acquisition envelope.
            amplitude (float): amplitude of the RF source in mV.
            attenuation (float): Attenuation of the source channel.
            startup_time_ns (float):
                startup time [ns] of the resonator. In pulsed and continuous mode the RF source is started
                `startup_time_ns` before acquisition..
            prolongation_ns (float):
                prolongation time [ns] of the RF source after acquisition end in pulsed mode.
            source_delay_ns (float):
                delay to be added to the source signal [ns].
        Note:
            The output specification depends on the driver.
            Qblox driver only supports module name with channel number(s).
        '''
        rf_source = resonator_rf_source(output=output, mode=mode,
                                        amplitude=amplitude,
                                        attenuation=attenuation,
                                        startup_time_ns=startup_time_ns,
                                        prolongation_ns=prolongation_ns,
                                        delay=source_delay_ns)
        self.digitizer_channels[channel_name].rf_source = rf_source

    def set_digitizer_iq_out(self, channel_name, iq_out):
        '''
        Enables/disables IQ output of digitizer channel.
        Args:
            channel_name (str): name of the channel.
            iq_out (bool): if True output I+Q data, else output I data only.
        '''
        self.digitizer_channels[channel_name].iq_out = iq_out

    def add_channel_delay(self, channel, delay):
        '''
        Adds to a channel a delay.
        The delay is added by adding points in front of the first sequence/or
        just after the last sequence. The first value of the sequence will be
        taken as an extentsion point.

        Args:
            channel (str) : channel name as defined in self.define_channel().
            delay (int): delay to be added to the channel (this may be a postive or negative number).
        '''
        if channel in self.awg_channels:
            self.awg_channels[channel].delay = delay
        elif channel in self.marker_channels:
            self.marker_channels[channel].delay = delay
        elif channel in self.digitizer_channels:
            self.digitizer_channels[channel].delay = delay
        else:
            raise ValueError(f"Channel delay error: Channel '{channel}' is not defined")

    def add_channel_compensation_limit(self, channel_name, limit):
        '''
        Sets voltage limits per channnel. They are used to make sure that the intregral of the total voltages is 0.
        Args:
            channel (str) : channel name as defined in self.define_channel().
            limit (tuple<float,float>) : lower/upper limit for DC compensation, e.g. (-100,500)
        '''
        if channel_name in self.awg_channels:
            self.awg_channels[channel_name].compensation_limits = limit
        else:
            raise ValueError(f"Channel compensation delay error: Channel '{channel_name}' is not defined")

    def add_channel_attenuation(self, channel_name, attenuation):
        '''
        Sets channel attenuation factor (AWG-to-DAC ratio).
        Args:
            channel_name (str) : channel name as defined in self.define_channel().
            attenuation (float) : attenuation factor
        '''
        if channel_name in self.awg_channels:
            self.awg_channels[channel_name].attenuation = attenuation
        else:
            raise ValueError(f"Channel '{channel_name}' is not defined")

    def add_channel_bias_T_compensation(self, channel_name, bias_T_RC_time):
        '''
        Sets the bias-T RC time for the bias-T compensation.
        Args:
            channel_name (str) : channel name as defined in self.define_channel().
            bias_T_RC_time (float) : RC time of bias-T
        '''
        if channel_name in self.awg_channels:
            self.awg_channels[channel_name].bias_T_RC_time = bias_T_RC_time
        else:
            raise ValueError(f"Channel '{channel_name}' is not defined")

    def add_channel_offset(self, channel_name, offset):
        '''
        Sets channel offset.
        Args:
            channel_name (str) : channel name as defined in define_channel().
            offset (float) : offset in mV.
        '''
        if channel_name in self.awg_channels:
            self.awg_channels[channel_name].offset = offset
        else:
            raise ValueError(f"Channel '{channel_name}' is not defined")

    # pylint: disable-next=too-many-arguments
    def define_iq_channel(
            self, name: str, i_name: str = "", q_name: str = "",
            i_neg_name: str = "", q_neg_name: str = "",
            marker_name: str = ""
    ) -> None:
        '''
        Defines a new IQ pair for IQ modulated MW pulse generation via qubit channels.

        For balanced IQ signals two additional channels may be specified for the
        negative image of I and Q signal.

        Args:
            i_name: awg channel name of the I component (+)
            q_name: awg channel name of the Q component (+)
            i_neg_name: [optional] awg channel name of the I component (-)
            q_neg_name: [optional] awg channel name of the Q component (-)
            marker_name: [optional] marker to use for pulse modulation.
        '''
        iq_channel = IQ_channel(name)
        self.IQ_channels[name] = iq_channel

        if i_name:
            iq_channel.add_awg_out_chan(i_name, "I", "+")
        if q_name:
            iq_channel.add_awg_out_chan(q_name, "Q", "+")
        if i_neg_name:
            iq_channel.add_awg_out_chan(i_neg_name, "I", "-")
        if q_neg_name:
            iq_channel.add_awg_out_chan(q_neg_name, "Q", "-")
        if marker_name:
            iq_channel.add_marker(marker_name)

    def set_iq_lo(self, iq_channel_name: str, lo: Union[Parameter, float]) -> None:
        '''
        Sets the LO frequency (parameter) for the IQ channel.
        Args:
            iq_channel_name: name of IQ channel
            lo: qcodes parameter for LO frequency or float value [Hz]
        '''
        self.IQ_channels[iq_channel_name].LO_parameter = lo

    def define_IQ_channel(self, name):
        print("WARNING: define_IQ_channel is deprecated use define_iq_channel")
        self.define_iq_channel(name)
        return self.IQ_channels[name]

    def define_qubit_channel(self, qubit_channel_name, IQ_channel_name,
                             resonance_frequency=None,
                             correction_phase=0.0, correction_gain=(1.0, 1.0)
                             ):
        """
        Creates a qubit channel on an IQ channel. A qubit channel keeps track of the
        qubit's phase required for coherent quantum gates.

        Args:
            qubit_channel_name (str) : channel name (e.g. q1)
            IQ_channel_name: name of the IQ pair.
            resonance_frequency (float) : frequency of the qubit when not driving and default for driving.
            correction_phase (float) : phase in rad added to Q component of IQ channel
            correction_gain (float,float) : correction of I and Q amplitude
        """
        iq_channel = self.IQ_channels[IQ_channel_name]
        qubit = QubitChannel(qubit_channel_name, resonance_frequency, iq_channel,
                             correction_phase, correction_gain)
        iq_channel.qubit_channels.append(qubit)
        self.qubit_channels[qubit_channel_name] = qubit

    def set_qubit_idle_frequency(self, qubit_channel_name, resonance_frequency):
        print('set_qubit_idle_frequency is deprecated. Use set_qubit_resonance_frequency')
        self.set_qubit_resonance_frequency(qubit_channel_name, resonance_frequency)

    def set_qubit_resonance_frequency(self, qubit_channel_name, resonance_frequency):
        self.qubit_channels[qubit_channel_name].resonance_frequency = resonance_frequency

    def set_qubit_correction_phase(self, qubit_channel_name, correction_phase):
        self.qubit_channels[qubit_channel_name].correction_phase = correction_phase

    def set_qubit_correction_gain(self, qubit_channel_name, correction_gain_I, correction_gain_Q):
        self.qubit_channels[qubit_channel_name].correction_gain = (correction_gain_I, correction_gain_Q)

    def set_channel_attenuations(self, attenuation_dict: Dict[str, float]):
        for channel, attenuation in attenuation_dict.items():
            if channel not in self.awg_channels:
                logger.info(f'Channel {channel} defined in hardware, but not in pulselib; skipping channel')
                continue
            self.awg_channels[channel].attenuation = attenuation

    def get_channel_attenuations(self) -> Dict[str, float]:
        return {c.name: c.attenuation for c in self.awg_channels.values()}

    def add_virtual_matrix(self, name,
                           real_gate_names,
                           virtual_gate_names,
                           matrix,
                           real2virtual=False,
                           filter_undefined=False,
                           keep_squared=False):
        self._virtual_matrices.add(
                name,
                real_gate_names,
                virtual_gate_names,
                matrix,
                real2virtual=real2virtual,
                filter_undefined=filter_undefined,
                keep_squared=keep_squared,
                awg_channels=list(self.awg_channels.keys())
                )

    def set_default_hw_schedule_creator(self, hw_schedule_creator):
        self._default_hw_schedule_creator = hw_schedule_creator

    def _create_M3202A_uploader(self):
        try:
            from pulse_lib.keysight.M3202A_uploader import M3202A_Uploader
        except (ImportError, OSError):
            logger.error('Import of Keysight M3202A uploader failed', exc_info=True)
            raise

        self.uploader = M3202A_Uploader(self.awg_devices, self.awg_channels,
                                        self.marker_channels, self.qubit_channels,
                                        self.digitizers, self.digitizer_channels)

    def _create_Tektronix5014_uploader(self):
        try:
            from pulse_lib.tektronix.tektronix5014_uploader import Tektronix5014_Uploader
        except ImportError:
            logger.error('Import of Tektronix uploader failed', exc_info=True)
            raise

        self.uploader = Tektronix5014_Uploader(self.awg_devices, self.digitizers,
                                               self.awg_channels, self.marker_channels,
                                               self.digitizer_markers, self.qubit_channels,
                                               self.digitizer_channels, self.awg_sync)

    def _old_Tektronix5014_message(self):
        raise Exception('''
        Pulselib Tektronix driver has changed in pulselib version 1.3.6.
        New driver: backend='Tektronix_5014'.
        ATTENTION:
            * Amplitude output has been corrected. It is 2x previous output. Correct attenuation per channel!!
            * Use sequence.play(release=False) to call play multiple times after a single upload.
            ''')

    def _create_KeysightQS_uploader(self):
        try:
            from pulse_lib.keysight.qs_uploader import QsUploader
        except ImportError:
            logger.error('Import of KeysightQS uploader failed', exc_info=True)
            raise

        self.uploader = QsUploader(self.awg_devices, self.awg_channels,
                                   self.marker_channels,
                                   self.IQ_channels, self.qubit_channels,
                                   self.digitizers, self.digitizer_channels)

    def _create_QbloxPulsar_uploader(self):
        try:
            from pulse_lib.qblox.pulsar_uploader import PulsarUploader
        except ImportError:
            logger.error('Import of QbloxPulsar uploader failed', exc_info=True)
            raise

        self.uploader = PulsarUploader(self.awg_devices, self.awg_channels,
                                       self.marker_channels,
                                       self.IQ_channels, self.qubit_channels,
                                       self.digitizers, self.digitizer_channels)
        # QRM is always controlled by pulselib
        self.configure_digitizer = True

    def finish_init(self):
        if self._backend in ["Keysight", "M3202A"]:
            self._create_M3202A_uploader()

        elif self._backend == "Tektronix5014":
            self._old_Tektronix5014_message()

        elif self._backend == "Tektronix_5014":
            self._create_Tektronix5014_uploader()

        elif self._backend == "Keysight_QS":
            self._create_KeysightQS_uploader()

        elif self._backend == "Qblox":
            self._create_QbloxPulsar_uploader()

        else:
            raise Exception(f'Unknown backend: {self._backend}')

    def mk_segment(self, name=None, sample_rate=None, hres=False):
        '''
        generate a new segment.
        Returns:
            segment (segment_container) : returns a container that contains all the previously defined gates.
        '''
        return segment_container(self.awg_channels.keys(), self.marker_channels.keys(),
                                 self._virtual_matrices, self.IQ_channels.values(),
                                 self.digitizer_channels.values(),
                                 name=name, sample_rate=sample_rate, hres=hres)

    def mk_sequence(self, segments):
        '''
        segments: list of segment_container.
        '''
        default_hw_schedule = None
        if self._default_hw_schedule_creator is not None:
            default_hw_schedule = self._default_hw_schedule_creator(self)
        seq_obj = sequencer(self.uploader, self.digitizer_channels, self.awg_channels,
                            default_hw_schedule=default_hw_schedule)
        seq_obj.add_sequence(segments)
        seq_obj.configure_digitizer = self.configure_digitizer
        return seq_obj

    def release_awg_memory(self, wait_idle=True):
        """
        Releases AWG waveform memory.
        Also flushes AWG queues.
        """
        if self._backend == "Tektronix_5014":
            self.uploader.release_all_awg_memory()
            return

        if self._backend not in ["Keysight", "Keysight_QS", "M3202A"]:
            logger.info(f'release_awg_memory() not implemented for {self._backend}')
            return

        if wait_idle:
            self.uploader.wait_until_AWG_idle()

        self.uploader.release_memory()

        for channel in self.awg_channels.values():
            awg = self.awg_devices[channel.awg_name]
            awg.awg_flush(channel.channel_number)

        self.uploader.release_all_awg_memory()

    def load_hardware(self, hardware):
        '''
        load virtual gates and attenuation via the harware class (used in qtt)

        Args:
            hardware (harware_parent) : harware class.
        '''
        try:
            from core_tools.drivers.hardware.hardware import hardware as hw_cls
        except Exception:
            logger.warning('old version of core_tools detected ..')

        try:
            new_hardware_class = isinstance(hardware, hw_cls)
        except Exception:
            new_hardware_class = False

        if new_hardware_class:
            for virtual_gate_set in hardware.virtual_gates:
                self.add_virtual_matrix(
                        virtual_gate_set.name,
                        virtual_gate_set.real_gate_names,
                        virtual_gate_set.virtual_gate_names,
                        virtual_gate_set.matrix,
                        real2virtual=True,
                        filter_undefined=True,
                        keep_squared=True)

            # Add all awg channels to mapping
            hardware.awg2dac_ratios.add(list(self.awg_channels.keys()))

            self.set_channel_attenuations(hardware.awg2dac_ratios)

        else:
            for virtual_gate_set in hardware.virtual_gates:
                self.add_virtual_matrix(
                        virtual_gate_set.name,
                        virtual_gate_set.real_gate_names,
                        virtual_gate_set.virtual_gate_names,
                        virtual_gate_set.virtual_gate_matrix,
                        real2virtual=True,
                        filter_undefined=True,
                        keep_squared=True)

            self.set_channel_attenuations(hardware.AWG_to_dac_conversion)

            # Add all awg channels to mapping
            sync = False
            for channel in self.awg_channels.values():
                if channel.name not in hardware.AWG_to_dac_conversion:
                    hardware.AWG_to_dac_conversion[channel.name] = channel.attenuation
                    sync = True
            if sync:
                hardware.sync_data()

    def _check_uniqueness_of_channel_name(self, channel_name):
        if len(channel_name) < 2:
            raise Exception(f"Channel name '{channel_name}' is too short. Min is 2")
        if len(channel_name) > 16:
            raise Exception(f"Channel name '{channel_name}' is too long. Max is 16.")
        if '{' in channel_name:
            raise Exception(f"Illegal channel name '{channel_name}'. Did you forget the f in front of the string?")
        if not re.match(r"^[A-Za-z][A-Za-z0-9_]*$", channel_name):
            raise Exception(f"Invalid channel name '{channel_name}'")

        if (channel_name in self.awg_channels
                or channel_name in self.marker_channels
                or channel_name in self.digitizer_channels
                or channel_name in self.qubit_channels):
            raise ValueError(f"duplicate declaration of the a channel/marker name '{channel_name}'.")
