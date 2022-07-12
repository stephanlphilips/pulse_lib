from dataclasses import dataclass
from typing import Optional, Union, List

from pulse_lib.segments.utility.looping import loop_obj

@dataclass
class AcquisitionConf:
    configure_digitizer: bool = False
    '''
    If true the digitizer will be configured by pulselib.
    '''

    t_measure: Union[None, float, loop_obj] = None
    '''
    measurement time in ns.
    If None it must be set in acquire()
    '''
    channels: Optional[List[str]] = None
    '''
    Channels to retrieve data from specified by name.
    If None it is defined by acquire()
    '''
    sample_rate: Optional[float] = None
    '''
    Sample rate of data in Hz. When not None, the data should not be averaged,
    but downsampled with specified rate. Useful for time traces and Elzerman readout.
    Downsampling uses block average.
    '''
    average_repetitions: bool = False
    '''
    Average acquisition data over the sequence repetitions.
    '''

    # TODO are the options needed?
    # options: Optional[Dict[str,Any]] = None
    # '''
    # Instrument specific options that will not be interpreted by pulse_lib.
    # Examples: sample_rate, mV_range,
    # '''

    # TODO?
    # * complex: (bool) = False ??
    #
    # * channel_map: (Dict[]) = None ??
    #   Or use virtual channel to define mappings?
    #
    # Generic, but doesn't allow mapping to hardware => use 2 mechanisms
    # add_derived_output('name', lambda d:abs(d['SD1'])
    # add_derived_output('name', lambda d:abs(d['SD1']-abs(d['SD2'])
