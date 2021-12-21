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
    sequencer_name : str = None
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
class digitizer_channel:
    '''
    Channel to retrieve the digitizer data from.

    NOTE:
        This channel does not specify the physical digitizer input channel.
        The digitizer can combine two physical inputs in one output buffer.
        It can also demodulate 1 physcial input to multipe output buffers.
    '''
    name: str
    module_name: str
    channel_number: int
    '''
    Channel number to *read* the data from.
    This is the number of the output buffer of the digitizer.
    '''
    iq_out: bool = False
    '''
    Return I/Q data in complex value. If False the imaginary component will be discarded.
    '''
    downsample_rate: Optional[float] = None
    '''
    When not None, the data should not be averaged, but downsampled with specified rate.
    Can be used for Elzerman readout.
    '''

    @property
    def channel_numbers(self):
        ''' Returns channel number in list.
        Utility method to simplify code that accepts classes digitizer_channel and digitizer_channel_iq
        '''
        return [self.channel_number]

@dataclass
class digitizer_channel_iq:
    '''
    I/Q channel pair consisting of 2 input channels that are not combined by the hardware.
    Pair is treated as one entity.
    Both channels are triggered simultaneously.
    Acquisition result is complex value.
    '''
    name: str
    module_name: str
    channel_numbers: List[int]
    phase : float = 0.0
    iq_out: bool = False
    '''
    Return I/Q data in complex value. If False the imaginary component will be discarded.
    '''
    downsample_rate: Optional[float] = None
    '''
    When not None, the data should not be averaged, but downsampled with specified rate.
    Can be used for Elzerman readout.
    '''

