from pulse_lib.tests.configurations.test_configuration import context

#%%
from qcodes import Parameter
from pulse_lib.fast_scan.qblox_fast_scans import fast_scan1D_param, fast_scan2D_param
from core_tools.sweeps.sweeps import do1D

class RfParameter(Parameter):
    def __init__(self, pulselib, digitizer_channel_name):
        super().__init__(
                name=f'{digitizer_channel_name}_RF',
                label=f'{digitizer_channel_name} resonator frequency',
                unit='Hz')
        self.channel = pulselib.digitizer_channels[digitizer_channel_name]

    def get_raw(self):
        return self.channel.frequency

    def set_raw(self, value):
        self.channel.frequency = value


class RfAmplitudeParameter(Parameter):
    def __init__(self, pulselib, digitizer_channel_name):
        super().__init__(
                name=f'{digitizer_channel_name}_amplitude',
                label=f'{digitizer_channel_name} resonator drive amplitude',
                unit='mV')
        self.channel = pulselib.digitizer_channels[digitizer_channel_name]

    def get_raw(self):
        return self.channel.rf_source.amplitude

    def set_raw(self, value):
        self.channel.rf_source.amplitude = value


class PhaseParameter(Parameter):
    def __init__(self, pulselib, digitizer_channel_name):
        super().__init__(
                name=f'{digitizer_channel_name} phase',
                label=f'{digitizer_channel_name} phasey',
                unit='degrees')
        self.channel = pulselib.digitizer_channels[digitizer_channel_name]

    def get_raw(self):
        return self.channel.phase

    def set_raw(self, value):
        self.channel.phase = value


def test_freq():
    pulse = context.init_pulselib(n_gates=2, n_sensors=2, rf_sources=True)

    rf_frequency = RfParameter(pulse, 'SD2')
    fast_scan_param = fast_scan1D_param(pulse, 'P1', 50.0, 21, 2_000, iq_mode='I+Q')
    ds = do1D(rf_frequency, 80e6, 120e6, 21, 0.0, fast_scan_param, name='frequency_search', reset_param=True).run()

    return ds

def test_ampl():
    pulse = context.init_pulselib(n_gates=2, n_sensors=2, rf_sources=True)

    rf_amplitude = RfAmplitudeParameter(pulse, 'SD2')
    fast_scan_param = fast_scan1D_param(pulse, 'P1', 50.0, 21, 2_000, iq_mode='I+Q')
    ds = do1D(rf_amplitude, 20.0, 200.0, 10, 0.0, fast_scan_param, name='amplitude_sweep', reset_param=True).run()

    return ds

#%%
if __name__ == '__main__':
    context.init_coretools()
    ds1 = test_freq()
    ds2 = test_ampl()
