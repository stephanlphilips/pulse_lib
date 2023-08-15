import math
import numpy as np
from qcodes import Instrument
# import M4i to add pyspcm to path. pyspcm is used by pulse-lib code.
from qcodes_contrib_drivers.drivers.Spectrum.M4i import M4i


class MockM4i(Instrument):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter('timeout', set_cmd=None, initial_value=10000)
        self.add_parameter('sample_rate', set_cmd=None, initial_value=100e6)
        self.add_parameter('segment_size', set_cmd=None, initial_value=32)
        self.add_parameter('enable_channels', set_cmd=None, initial_value=0)
        self.add_parameter('data_memory_size', set_cmd=None, initial_value=0)
        self.add_parameter('pretrigger_memory_size', set_cmd=None, initial_value=16)

    def get_idn(self):
        return dict(vendor='Pulselib', model=type(self).__name__, serial='', firmware='')

    def start_triggered(self):
        pass

    def trigger_or_mask(self, value):
        pass

    def box_averages(self, value):
        pass

    def setup_multi_recording(self, seg_size, n_triggers, boxcar_average):
        self.segment_size(math.ceil(seg_size/16)*16 + 16)
        self.data_memory_size(self.segment_size()*n_triggers)

    def get_data(self):
        channels = self.enable_channels()
        n_ch = 0
        while channels:
            if channels & 1:
                n_ch += 1
            channels >>= 1
        n_samples = self.data_memory_size()
        res = np.empty((n_ch, n_samples))
        for i in range(n_ch):
            res[i] = np.linspace(0, 1.0, n_samples)
        return res
