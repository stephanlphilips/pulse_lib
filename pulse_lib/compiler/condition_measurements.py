from collections.abc import Iterable
from collections import defaultdict
import logging
import copy

import numpy as np

from pulse_lib.segments.segment_measurements import measurement_acquisition
from pulse_lib.segments.conditional_segment import conditional_segment


logger = logging.getLogger(__name__)

'''
ConditionSegment -> Acquisition
Acquisition: Segment#, end-time

Feedback event:
    * measurement: channel
    * event set time: measurement end-time
    * event clear time: condition end-time
    * conditions

* Qblox:
    * latch-reset on segment start? latch idle time.
    * if multiple measurements on sensor: latch-enable interval / latch-disable interval.
* Keysight:
    * pxi-trigger assignment
    * Add support for multiple pxi-triggers
    * Add pxi-triggers assignment to configuration: single condition could use high bit.
'''

class FeedbackEvent:
    def __init__(self, measurement, end_times):
        self.measurement = measurement
        self._set_times = end_times
        self._apply_times = np.full(1, np.nan)
        self._reset_times = end_times

    @property
    def set_times(self):
        return self._set_times

    @property
    def apply_times(self):
        return self._apply_times

    @property
    def reset_times(self):
        return self._reset_times

    def add_condition(self, condition, start_times, end_times):
        self._apply_times = np.fmin(self._apply_times, start_times)
        self._reset_times = np.fmax(self._reset_times, end_times)



class ConditionMeasurements:
    def __init__(self, measurements_description, uploader, max_awg_to_dig_delay):
        self._md = measurements_description
        self._supports_conditionals = getattr(uploader, 'supports_conditionals', False)
        if self._supports_conditionals:
            self.min_feedback_time = uploader.get_roundtrip_latency() + max_awg_to_dig_delay
        self._measurements = {}
        self._feedback_events = {}
        self._end_times = {}
        self._n_segments = 0
        self._acquisition_count = defaultdict(int)
        self._channel_measurements = defaultdict(list)
        self._conditional_measurements = {}

    def add_segment(self, segment, seg_start_times):
        self._n_segments += 1
        self._add_measurements(segment, seg_start_times)

        if not isinstance(segment, conditional_segment):
            return

        if not self._supports_conditionals:
            raise Exception(f'Backend does not support conditional segments')

        condition = segment.condition
        seg_end = seg_start_times + segment.total_time

        # Lookup acquistions for condition
        acquisition_names = self._get_acquisition_names(condition)
        logger.info(f'segment {self._n_segments-1} conditional on: {acquisition_names}')

        # Add condition to feedback event
        measurements = []
        for name in acquisition_names:
            # NOTE: lastest measurement with this name before acquisition
            try:
                m = self._measurements[name]
            except KeyError:
                raise Exception(f'measurement {name} not found before condition')
            measurements.append(m)
            try:
                fb = self._feedback_events[id(m)]
            except KeyError:
                fb = FeedbackEvent(m, self._end_times[id(m)])
                self._feedback_events[id(m)] = fb
            fb.add_condition(condition, seg_start_times, seg_end)

        self._conditional_measurements[id(segment)] = measurements

    def _add_measurements(self, segment, seg_start_times):
        if isinstance(segment, conditional_segment):
            # Conditional branches must all have the same measurements.
            # use 1st branch of conditional segment.
            segment = segment.branches[0]
        for measurement in segment.measurements:
            if isinstance(measurement, measurement_acquisition):
                m = copy.copy(measurement)
                channel_acquisitions = self._channel_measurements[m.acquisition_channel]
                m.index += len(channel_acquisitions)
                channel_acquisitions.append(m)

                acq_channel = segment[m.acquisition_channel]
                end_times = np.zeros(seg_start_times.shape)
                for index in np.ndindex(seg_start_times.shape):
                    acq_data = acq_channel._get_data_all_at(index).data[measurement.index]
                    t_measure = acq_data.t_measure if acq_data.t_measure is not None else 0
                    end_times[index] = seg_start_times[index] + acq_data.start + t_measure
                self._end_times[id(m)] = end_times
            else:
                m = measurement
            self._measurements[m.name] = m

    @property
    def feedback_events(self):
        return self._feedback_events

    @property
    def measurement_acquisitions(self):
        return self._channel_measurements

    def get_end_time(self, measurement, index):
        return self._end_times[id(measurement)][tuple(index)]

    def get_measurements(self, conditional_segment):
        return self._conditional_measurements[id(conditional_segment)]

    def check_feedback_timing(self):
        if not self._supports_conditionals:
            return

        required_time = self.min_feedback_time

        for fb in self._feedback_events.values():
            margin = fb.apply_times - fb.set_times - required_time
            if np.min(margin) < 0:
                raise Exception(f'Insufficient time between measurement {fb.measurement.name} and condition')

    def _get_acquisition_names(self, condition):
        if isinstance(condition, Iterable):
            acquisition_names = set()
            for ref in condition:
                acquisition_names.update(ref.keys)
            return list(acquisition_names)
        return list(condition.keys)

