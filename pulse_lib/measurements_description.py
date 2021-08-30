import copy
from .segments.segment_measurements import measurement_acquisition
from .segments.conditional_segment import conditional_segment

class measurements_description:
    def __init__(self, digitizer_channels):
        self.measurements = []
        self.acquisitions = {}
        self.acquisition_count = {}
        self.digitizer_channels = digitizer_channels
        self.end_times = {}

    def add_segment(self, segment, seg_start_times):
        if isinstance(segment, conditional_segment):
            # Currently conditional branches must all have the same measurement.
            # use 1st branch of conditional segment.
            segment = segment.branches[0]
        for measurement in segment.measurements:
            if isinstance(measurement, measurement_acquisition):
                m = copy.copy(measurement)
                acquisition_count = self.acquisition_count.setdefault(m.acquisition_channel, 0)
                m.index += acquisition_count

                times = seg_start_times.flatten()
                acq_channel = segment[m.acquisition_channel]
                for i in range(len(times)):
                    acq_data = acq_channel.pulse_data_all.flat[i].data[measurement.index]
                    times[i] += acq_data.start + acq_data.t_measure
                self.end_times[m.name] = times.reshape(seg_start_times.shape)
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

    def describe(self):
        print('Measurements:')
        for m in self.measurements:
            print(m)
        for channel_name, acq in self.acquisitions.items():
            print(f"Acquisitions '{channel_name}':")
            d  = acq.flat[0]
            print(d.data)
