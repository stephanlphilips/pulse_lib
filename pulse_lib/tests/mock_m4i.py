import math
import numpy as np
from qcodes import Instrument
# import M4i to add pyspcm to path. pyspcm is used by pulse-lib code.
from qcodes_contrib_drivers.drivers.Spectrum.M4i import M4i
import pyspcm


class MockM4i(Instrument):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter('clock_mode', set_cmd=None, initial_value=pyspcm.SPC_CM_INTPLL)
        self.add_parameter('reference_clock', set_cmd=None, initial_value=10_000_000)
        self.add_parameter('timeout', set_cmd=None, initial_value=10000)
        self.add_parameter('sample_rate', set_cmd=None, initial_value=100e6)
        self.add_parameter('segment_size', set_cmd=None, initial_value=32)
        self.add_parameter('enable_channels', set_cmd=None, initial_value=0)
        self.add_parameter('data_memory_size', set_cmd=None, initial_value=0)
        self.add_parameter('pretrigger_memory_size', set_cmd=None, initial_value=16)
        self._data = {}
        self._n_triggers = 0

    def get_idn(self):
        return dict(vendor='Pulselib', model=type(self).__name__, serial='', firmware='')

    def start_triggered(self):
        pass

    def trigger_or_mask(self, value):
        pass

    def box_averages(self, value):
        pass

    def initialize_channels(self, channels=None, mV_range=1000, input_path=0,
                            termination=0, coupling=0, compensation=None,
                            memsize=2**12, pretrigger_memsize=16,
                            lp_filter=None):
        pass

    def set_ext0_OR_trigger_settings(self, trig_mode=1, termination=0, coupling=0,
                                     level0=1600, level1=None):
        pass

    def setup_multi_recording(self, seg_size, n_triggers, boxcar_average):
        pretrigger = self.pretrigger_memory_size()
        self.segment_size(math.ceil(seg_size/16)*16 + pretrigger)
        self.data_memory_size(self.segment_size()*n_triggers)
        self._n_triggers = n_triggers

    def set_data(self, ch_num, data):
        self._data[ch_num] = data

    def get_data(self):
        pretrigger = self.pretrigger_memory_size()
        channels = self.enable_channels()
        ch_nums = []
        for ch_num in range(4):
            if channels & (1 << ch_num):
                ch_nums.append(ch_num)
        n_ch = len(ch_nums)
        n_samples = self.data_memory_size()
        res = np.full((n_ch, n_samples), np.nan)
        for i in range(n_ch):
            data = self._data.get(ch_nums[i], None)
            if data is None:
                res[i] = np.linspace(0, 1.0, n_samples)
            else:
                mem_data = np.full((self._n_triggers, self.segment_size()), np.nan)
                d = data.reshape(self._n_triggers, -1)
                mem_data[:,pretrigger:pretrigger+d.shape[1]] = d
                res[i] = mem_data.flatten()
        return res
