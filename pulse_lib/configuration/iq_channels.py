from dataclasses import dataclass, field
from typing import List, Union, Tuple

from qcodes.instrument.parameter import Parameter

@dataclass
class IQ_out_channel_info:
    awg_channel_name: str
    # I or Q component
    IQ_comp: str
    # make the negative of positive image of the signal (*-1)
    image: str


@dataclass
class QubitChannel:
    channel_name : str
    reference_frequency : float
    iq_channel: 'IQ_channel'
    correction_phase: float = 0.0
    correction_gain: Tuple[float] = (1.0, 1.0)


@dataclass
class IQ_channel:
    name: str
    qubit_channels: List[QubitChannel] = field(default_factory=list)
    IQ_out_channels: List[IQ_out_channel_info] = field(default_factory=list)
    marker_channels: List[str] = field(default_factory=list)
    LO_parameter: Union[None, float, Parameter] = None

    @property
    def LO(self):
        """
        get LO frequency of the MW source
        """
        if isinstance(self.LO_parameter, float):
            return self.LO_parameter
        elif isinstance(self.LO_parameter, Parameter):
            return self.LO_parameter.cache.get()
        else:
            raise ValueError("Local oscillator not set in the IQ_channel.")

