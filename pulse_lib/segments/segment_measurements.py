"""
Measurement channel implementation.
"""
from dataclasses import dataclass
from typing import Optional, Callable

import numpy as np

from .utility.measurement_ref import MeasurementExpressionBase, MeasurementRef

@dataclass
class measurement_base:
    name: Optional[str]
    accept_if: Optional[bool]

@dataclass
class measurement_acquisition(measurement_base):
    acquisition_channel: str
    index: int
    threshold: Optional[float] = None
    zero_on_high: bool = False
    ref: Optional[MeasurementRef] = None
    t_measure: Optional[float] = None
    n_repeat: Optional[int] = None
    interval: Optional[float] = None # [ns]
    n_samples: Optional[int] = None
    '''  Number of samples when using time traces. Value set by sequencer when downsampling. '''
    data_offset: int = 0
    ''' Offset of data in acquired channel data. '''
    aggregate_func: Callable[[np.ndarray], np.ndarray] = None
    '''
    Function aggregating data on time axis to new value.
    '''

    @property
    def has_threshold(self):
        return self.threshold is not None


@dataclass
class measurement_expression(measurement_base):
    expression: Optional[MeasurementExpressionBase] = None

# NOTE: this segment has no dimensions!
class segment_measurements:
    def __init__(self):
        self._measurements = []

    @property
    def measurements(self):
        return self._measurements

    def add_acquisition(self, channel:str, index:int,
                        t_measure:Optional[float],
                        threshold:Optional[float],
                        zero_on_high=False,
                        ref:MeasurementRef=None,
                        accept_if=None,
                        n_repeat=None,
                        interval=None):
        if ref is None:
            name = None
        elif isinstance(ref, str):
            name = ref
        else:
            name = ref.name
        self._measurements.append(measurement_acquisition(name, accept_if, channel, index,
                                                          threshold, zero_on_high, ref,
                                                          t_measure, n_repeat=n_repeat,
                                                          interval=interval))

    def add_expression(self, expression:MeasurementExpressionBase, accept_if=None, name:str=None):
        if name is None:
            name = f'<unnamed> {expression}'
        self._measurements.append(measurement_expression(name, accept_if, expression))

    def __getitem__(self, item):
        raise NotImplementedError()

    def __add__(self, other):
        if (len(self._measurements) > 0
            or len(other._measurements) > 0):
            raise Exception(f'Measurements cannot (yet) be combined')
        return self


