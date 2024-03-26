from pulse_lib.tests.configurations.test_configuration import context

# %%
import numpy as np
from pulse_lib.scan.read_input import read_channels
from core_tools.sweeps.sweeps import do1D


def test_freq():
    pulse = context.init_pulselib(n_gates=2, n_sensors=2, rf_sources=True)

    rf_frequency = pulse.rf_params['SD1'].frequency
    meas_param = read_channels(pulse, 2_000, channels=['SD1'], iq_mode='I+Q')
    ds = do1D(rf_frequency, 80e6, 120e6, 21, 0.0, meas_param,
              name='frequency_search', reset_param=True).run()

    return ds


def test_ampl():
    pulse = context.init_pulselib(n_gates=2, n_sensors=2, rf_sources=True)

    rf_amplitude = pulse.rf_params['SD1'].source_amplitude
    meas_param = read_channels(pulse, 2_000, channels=['SD1'], iq_mode='I+Q')
    ds = do1D(rf_amplitude, 20.0, 200.0, 10, 0.0, meas_param, name='amplitude_sweep', reset_param=True).run()

    return ds


def test_phase():
    pulse = context.init_pulselib(n_gates=2, n_sensors=2, rf_sources=True)

    rf_phase = pulse.rf_params['SD1'].phase
    meas_param = read_channels(pulse, 2_000, channels=['SD1'], iq_mode='I+Q')
    ds = do1D(rf_phase, 0.0, 2*np.pi, 20, 0.0, meas_param, name='phase_sweep', reset_param=True).run()

    return ds


# %%
if __name__ == '__main__':
    context.init_coretools()
    ds1 = test_freq()
    ds2 = test_ampl()
    ds3 = test_phase()
