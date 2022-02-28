from dataclasses import dataclass
from typing import Optional, Union, List, Dict, Any

from pulse_lib.segments.utility.looping import loop_obj

@dataclass
class AcquisitionConf:
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
    downsample_rate: Optional[float] = None
    '''
    Downsampled rate in Hz. When not None, the data should not be averaged,
    but downsampled with specified rate. Useful for Elzerman readout.
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
    # * add indexer for channel specific configuration: t_measure, downsample_rate
    #
    # * complex: (bool) = False ??
    #
    # * channel_map: (Dict[]) = None ??
    #   Or use virtual channel to define mappings?
    #
    # Generic, but doesn't allow mapping to hardware => use 2 mechanisms
    # add_derived_output('name', lambda d:abs(d['SD1'])
    # add_derived_output('name', lambda d:abs(d['SD1']-abs(d['SD2'])
