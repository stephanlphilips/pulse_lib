from typing import List, Tuple
import logging
import numpy as np
from pulse_lib.segments.utility.looping import loop_obj
from pulse_lib.configuration.iq_channels import FrequencyUndefined

logger = logging.getLogger(__name__)

def get_iq_nco_idle_frequency(job, qubit_channel, index):
    '''
    Returns IQ / NCO frequency to use between pulses.
    This frequency is required for coherent driving of the qubit.

    The NCO frequency is derived from resonance frequency and LO frequency.
    The resonance frequency is retrieved from sequence or pulse_lib object.
    The former overrules the latter.

    The NCO frequency will be set to 0.0 if the qubit frequency is set to FrequencyUndefined.
    '''
    try:
        frequency = job.qubit_resonance_frequencies[qubit_channel.channel_name]
        if isinstance(frequency, loop_obj):
            frequency = frequency.at(index)
    except:
        frequency = qubit_channel.resonance_frequency
    if frequency is FrequencyUndefined:
        return 0.0
    if frequency is None:
        return None
    return frequency - qubit_channel.iq_channel.LO

def merge_markers(marker_name, marker_deltas, marker_value=1, min_off_ns=10) -> List[Tuple[int,int]]:
    '''
    Merge overlapping markers.

    Args:
        marker_name (str): name of marker used for logging.
        marker_deltas (List[Tuple(int, int)]): list with marker time and marker start (+1) or stop (-1).
        marker_value (int): step to add for marker start.
        min_off_ns (int): minimum time marker is off

    Returns:
        Sorted list with tuples of time and marker delta.
    '''
    res = []
    s = 0
    t_off = None
    for t,step in sorted(marker_deltas):
        s += step
        if s < 0:
            logger.error(f'Marker error {marker_name} at {t} ns')
        if s == 1 and step == +1:
            t_on = int(t)
            if t_off is not None and t_on - t_off < min_off_ns:
                # remove last t_off.
                res.pop()
            else:
                res.append((t_on, +marker_value))
        if s == 0 and step == -1:
            t_off = int(t)
            res.append((t_off, -marker_value))

    return res

