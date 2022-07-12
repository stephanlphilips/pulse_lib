import copy
import numpy as np
from .segments.segment_measurements import measurement_acquisition
from .segments.conditional_segment import conditional_segment

class measurements_description:
    def __init__(self, digitizer_channels):
        self.measurement_names = []
        self.measurements = []
        self.acquisitions = {}
        self.acquisition_count = {}
        self.channel_data_offset = {}
        self.digitizer_channels = digitizer_channels
        self.end_times = {}

    def add_segment(self, segment, seg_start_times):
        if isinstance(segment, conditional_segment):
            # Conditional branches must all have the same measurements.
            # use 1st branch of conditional segment.
            segment = segment.branches[0]
        for measurement in segment.measurements:
            if isinstance(measurement, measurement_acquisition):
                m = copy.copy(measurement)
                acquisition_count = self.acquisition_count.setdefault(m.acquisition_channel, 0)
                data_offset = self.channel_data_offset.setdefault(m.acquisition_channel, 0)
                m.index += acquisition_count
                m.data_offset = data_offset
                n_samples = m.n_samples if m.n_samples is not None else 1
                self.channel_data_offset[m.acquisition_channel] = data_offset + n_samples

                if m.name is None:
                    m.name = f'{m.acquisition_channel}_{m.index+1}'
                if m.name in self.measurement_names:
                    raise Exception(f'Duplicate measurement name: {m.name}')
                self.measurement_names.append(m.name)

                acq_channel = segment[m.acquisition_channel]
                end_times = seg_start_times.copy()
                for index in np.ndindex(seg_start_times.shape):
                    acq_data = acq_channel._get_data_all_at(index).data[measurement.index]
                    t_measure = acq_data.t_measure if acq_data.t_measure is not None else 0
                    end_times[index] += acq_data.start + t_measure
                self.end_times[m.name] = end_times
            else:
                m = measurement
            self.measurements.append(m)

        for channel_name, data in segment.acquisitions.items():
            acquisition_count = self.acquisition_count.setdefault(channel_name, 0)
            self.acquisition_count[channel_name] += len(data.flat[0].data)
            if channel_name not in self.acquisitions:
                self.acquisitions[channel_name] = data
            else:
                self.acquisitions[channel_name] = self.acquisitions[channel_name] + data

    def add_HVI_variables(self, HVI_variables):
        dig_triggers = []
        for v in HVI_variables.flat[0].HVI_markers:
            if v == 'dig_wait' or v.startswith('dig_trigger'):
                dig_triggers.append(v)

        if len(dig_triggers) and len(self.measurements):
            raise Exception('Cannot combine `acquire()` with HVI var `dig_trigger` or `dig_wait`')

        for index,trigger in enumerate(dig_triggers):
            for channel_name in self.digitizer_channels:
                name = f'{channel_name}_tr{index}'
                m = measurement_acquisition(name, False, channel_name, index)
                self.measurements.append(m)
                self.measurement_names.append(name)
                self.acquisition_count[channel_name] += 1

    def describe(self):
        print('Measurements:')
        for m in self.measurements:
            print(m)
        for channel_name, acq in self.acquisitions.items():
            print(f"Acquisitions '{channel_name}':")
            d  = acq.flat[0]
            print(d.data)

