from typing import Tuple
from dataclasses import dataclass


@dataclass
class awg_channel:
    name: str
    awg_name: str
    channel_number: int
    delay: float = 0 # ns
    attenuation: float = 1.0
    compensation_limits: Tuple[float, float] = (0,0)

@dataclass
class marker_channel:
    name: str
    module_name: str # could be AWG or digitizer
    channel_number: int # use 0 for Keysight trigger out channel
    setup_ns: float
    hold_ns: float
    amplitude: float = 1000
    invert: bool = False

