"""
Measurement channel implementation.
"""
from dataclasses import dataclass
from typing import Optional

from .utility.measurement_ref import MeasurementExpressionBase, MeasurementRef

@dataclass
class measurement_base:
    name: str
    accept_if: Optional[bool]

@dataclass
class measurement_acquisition(measurement_base):
    acquisition_channel: str
    index: int
    has_threshold: bool
    ref: Optional[MeasurementRef] = None

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

    def add_acquisition(self, channel:str, index:int, has_threshold:bool,
                        ref:MeasurementRef=None, accept_if=None):
        if ref is None:
            name = f'<unnamed> {channel},{index}'
        else:
            name = ref.name
        self._measurements.append(measurement_acquisition(name, accept_if, channel, index, has_threshold, ref))

    def add_expression(self, expression:MeasurementExpressionBase, accept_if=None, name:str=None):
        if name is None:
            name = f'<unnamed> {expression}'
        self._measurements.append(measurement_expression(name, accept_if, expression))

    def __getitem__(self, item):
        raise NotImplementedError()

    def __copy__(self):
        raise NotImplementedError()

