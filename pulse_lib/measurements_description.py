import copy
from .segments.segment_measurements import measurement_acquisition

class measurements_description:
    def __init__(self, digitizer_channels):
        self.measurements = []
        self.acquisitions = {}
        self.acquisition_count = {}
        self.digitizer_channels = digitizer_channels

    def add_segment(self, segment):

        for measurement in segment.measurements:
            if isinstance(measurement, measurement_acquisition):
                m = copy.copy(measurement)
                acquisition_count = self.acquisition_count.setdefault(m.acquisition_channel, 0)
                m.index += acquisition_count
            else:
                m = measurement
            self.measurements.append(m)

        for channel_name in segment.acquisitions:
            acquisition_count = self.acquisition_count.setdefault(channel_name, 0)
            self.acquisition_count[channel_name] += 1
            if channel_name not in self.acquisitions:
                self.acquisitions[channel_name] = segment[channel_name].data
            else:
                self.acquisitions[channel_name] = self.acquisitions[channel_name] + segment[channel_name].data

    def describe(self):
        print('Measurements:')
        for m in self.measurements:
            print(m)
        for channel_name, acq in self.acquisitions.items():
            print(f"Acquisitions '{channel_name}':")
            d  = acq.flat[0]
            print(d.data)
