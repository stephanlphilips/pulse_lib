from typing import Tuple, Optional, Union, List
from dataclasses import dataclass

@dataclass
class awg_channel:
    name: str
    awg_name: str
    channel_number: int
    amplitude: Optional[float] = None
    delay: float = 0 # ns
    attenuation: float = 1.0
    compensation_limits: Tuple[float, float] = (0,0)
    bias_T_RC_time: Optional[float] = None
    offset: Optional[float] = None # mV

@dataclass
class marker_channel:
    name: str
    module_name: str # could be AWG or digitizer
    channel_number: Union[int,Tuple[int,int]]
    '''
    Keysight: 0 = trigger out channel, 1...4 = analogue channel
    Tektronix: tuple = (channel,marker number), int = analogue channel
    Qblox: 0...3 = marker out.
    '''
    setup_ns: float
    hold_ns: float
    amplitude: float = 1000
    invert: bool = False
    delay: float = 0 # ns
    sequencer_name : Optional[str] = None
    '''
    Qblox only: name of qubit, awg or digitizer channel to use for sequencing
    '''

# NOTES on digitizer configuration options for M3102A FPGA
#  * Input: I/Q demodulated (external demodulation) with pairing in FPGA
#    Output: I/Q, 2 channels
#    use digitizer_channel_iq and optionally set phase and iq_out
#    digitizer mode: NORMAL or AVERAGING
#    measurement_converter applies phase and generates 1 or 2 raw data outputs depending on iq_out
#
#  * Input: I/Q demodulated (external demodulation) with I/Q pairing in FPGA
#    Output: I/Q, 1 channel, real or complex valued output
#    use digitizer_channel optionally set iq_out=True
#    digitizer mode: IQ_INPUT_SHIFTED_IQ_OUT or IQ_INPUT_SHIFTED_I_ONLY
#    set phase in digitizer
#    measurement_converter generates 1 or 2 raw data outputs depending on iq_out
#
#  * Input: modulated signal, I/Q demodulation in FPGA
#    Output: I/Q, 1 channel, real or complex valued output
#    use digitizer_channel optionally set iq_out=True
#    digitizer mode: IQ_DEMODULATION or IQ_DEMOD_I_ONLY
#    set frequency and phase in digitizer
#    measurement_converter generates 1 or 2 raw data outputs depending on iq_out


@dataclass
class resonator_rf_source:
    '''
    RF source for resonator used with digitizer channel.
    The resonator will be driven with the frequency specified for the digitizer
    channel and dependent on the mode can be enabled synchronous with acquisitions.
    '''
    output: Union[str, Tuple[str,int], Tuple[str,List[int]]]
    '''
    output: one of the following:
        (str) name of awg_channel
        (Tuple[str, int]) name of module and channel number
        (Tuple[str, List[int]]) name of module and channel numbers
    '''
    mode: str = 'pulsed'
    '''
    'continuous', 'pulsed', 'shaped'
    '''
    amplitude: float = 0.0
    '''
    amplitude of the RF source in mV.
    '''
    trigger_offset_ns: float = 0.0
    '''
    offset in [ns] for pulsed and shaped RF source enabling.
    '''
    attenuation : float = 1.0
    '''
    Attenuation of the source channel.
    '''


@dataclass
class digitizer_channel:
    '''
    Channel to retrieve the digitizer data from.

    If multiple channel numbers are specified, than the acquisition for these
    channels is performed simultaneously.

    NOTE:
        This channel does not specify the physical digitizer input channel.
        The digitizer can combine two physical inputs in one output buffer.
        It can also demodulate 1 physcial input to multipe output buffers.
    '''
    name: str
    module_name: str
    channel_numbers: List[int]
    '''
    Channel number to *read* the data from.
    This is the number of the output buffer of the digitizer.
    '''
    # @@@ TODO change to 'data_mode': 'Complex' or 'Real' or 'I+Q'  or 'Split'??
    iq_out: bool = False
    '''
    Return I/Q data in complex value. If False the imaginary component will be discarded.
    '''
    phase : float = 0.0
    '''
    Phase shift after iq demodulation
    '''
    iq_input: bool = False
    '''
    Input consists of 2 channels, the demodulated I and Q.
    '''
    frequency: Union[None, float] = None
    '''
    demodulation frequency.
    '''
    rf_source: resonator_rf_source = None
    '''
    Optional rf_source to generate the resonator drive signal.
    '''

    def __post_init__(self):
        n_ch = len(self.channel_numbers)
        if self.iq_input and n_ch != 2:
            raise Exception(f'Channel {self.name} specified iq_input, but has {n_ch} channels')

    @property
    def channel_number(self):
        ''' Returns channel number if there is only 1 input channel.
        '''
        if len(self.channel_numbers) != 1:
            raise Exception(f'channel {self.name} has more than 1 channel')
        return self.channel_numbers[0]
