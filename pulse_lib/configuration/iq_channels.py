from dataclasses import dataclass, field
from typing import List, Union, Tuple, Optional

from qcodes.instrument.parameter import Parameter

@dataclass
class IQ_out_channel_info:
    awg_channel_name: str
    # I or Q component
    IQ_comp: str
    # make the negative of positive image of the signal (*-1)
    image: str

FrequencyUndefined = 'FrequencyUndefined'

@dataclass
class QubitChannel:
    channel_name : str
    resonance_frequency : Union[float, None, str]
    ''' qubit resonance frequency.
    None is not set.
    'UndefinedFrequency' implies non-coherent pulses using NCO frequency = 0.0
    '''
    iq_channel: 'IQ_channel'
    correction_phase: Optional[float] = None
    correction_gain: Optional[Tuple[float]] = None

    @property
    def reference_frequency(self):
        print('reference_frequency is deprecated')
        return self.resonance_frequency

    @reference_frequency.setter
    def reference_frequency(self, value):
        print('reference_frequency is deprecated')
        self.resonance_frequency = value

@dataclass
class IQ_channel:
    name: str
    qubit_channels: List[QubitChannel] = field(default_factory=list)
    IQ_out_channels: List[IQ_out_channel_info] = field(default_factory=list)
    marker_channels: List[str] = field(default_factory=list)
    LO_parameter: Union[None, float, int, Parameter] = None

    @property
    def LO(self):
        """
        get LO frequency of the MW source
        """
        if isinstance(self.LO_parameter, (float, int)):
            return self.LO_parameter
        elif isinstance(self.LO_parameter, Parameter):
            return self.LO_parameter.cache.get()
        else:
            raise ValueError("Local oscillator not set in the IQ_channel.")

    def add_awg_out_chan(self, awg_channel_name, IQ_comp, image = "+"):
        """
        AWG output channel for I or Q component.
        Args:
            awg_channel_name (str) : name of the channel in the AWG used to output
            IQ_comp (str) : "I" or "Q" singal that needs to be generated
            image (str) : "+" or "-", specify only when differential inputs are needed.
        """
        if IQ_comp not in ["I", "Q"]:
            raise ValueError(f"IQ component must be 'I' or 'Q', not '{IQ_comp}'")

        if image not in ["+", "-"]:
            raise ValueError(f"The image of the IQ signal must be '+' or '-', not '{image}'")

        for iq_out_ch in self.IQ_out_channels:
            if iq_out_ch.IQ_comp == IQ_comp and iq_out_ch.image == image:
                raise ValueError(f"Component {IQ_comp}{image} already defined for {self.name}")

        self.IQ_out_channels.append(IQ_out_channel_info(awg_channel_name, IQ_comp, image))

    def add_marker(self, marker_channel_name):
        """
        Channel for in phase information of the IQ channel (postive image)
        Args:
            marker_channel_name (str) : name of the channel in the AWG used to output
        """
        self.marker_channels.append(marker_channel_name)
