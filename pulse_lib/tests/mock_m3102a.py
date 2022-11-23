import numpy as np
from dataclasses import dataclass
from qcodes.instrument.base import Instrument

class MockM3102A(Instrument):

    def __init__(self, name, chassis, slot):
        super().__init__(name)

        self.chassis = chassis
        self.slot = slot

        self.measure = ChannelData()

    def get_idn(self):
        return dict(vendor='Pulselib', model=type(self).__name__, serial='', firmware='')

    def chassis_number(self):
        return self._chassis_numnber

    def slot_number(self):
        return self._slot_number

    def set_operating_mode(self, value):
        pass

    def set_acquisition_mode(self, value):
        pass

    def set_active_channels(self, channel_list):
        self.measure._active_channels = set(channel_list)

    @property
    def active_channels(self):
        return self.measure._active_channels

    def get_samples_per_measurement(self, t_measure, sample_rate):
        if sample_rate is None:
            return 1

        if sample_rate > 100e6:
            return int(t_measure*1e-9*sample_rate)

        downsampling_factor = int(max(1, round(100e6 / sample_rate)))
        t_downsampling = downsampling_factor * 10
        return max(1, round(t_measure/t_downsampling))

    def set_daq_settings(self, channel, n_cycles, t_measure, downsampled_rate=None):
        self.measure._active_channels.add(channel)
        properties = self.measure._ch_properties[channel]
        properties.n_cycles = n_cycles
        properties.t_measure = t_measure
        properties.samples_per_cycle = self.get_samples_per_measurement(t_measure, downsampled_rate)

    def set_data(self, channel, data):
        self.measure._data[channel] = data

@dataclass
class ChannelProperties:
    n_cycles: int = 1
    t_measure: int = 10
    samples_per_cycle: int = 1

class ChannelData:
    def __init__(self):
        self._active_channels = set([1,2,3,4])
        self._data = {i:None for i in self._active_channels}
        self._ch_properties = {i:ChannelProperties() for i in self._active_channels}

    def get_data(self):
        result = []
        for i in sorted(self._active_channels):
            properties = self._ch_properties[i]
            n_samples = properties.n_cycles * properties.samples_per_cycle
            data = self._data[i]
            if data is None:
                data = np.arange(n_samples)
            else:
                if len(data) != n_samples:
                    raise Exception("Length of provided data doesn't match. "
                                    f"Expected {n_samples}, got {len(data)}")

            result.append(data)
        return result

