from pulse_lib.tests.configurations.test_configuration import context

#%%
from pulse_lib.fast_scan.qblox_fast_scans import fast_scan1D_param
from core_tools.sweeps.sweeps import do1D


def test_freq():
    pulse = context.init_pulselib(n_gates=2, n_sensors=2, rf_sources=True)

    rf_frequency = pulse.rf_params['SD2'].frequency
    fast_scan_param = fast_scan1D_param(pulse, 'P1', 50.0, 21, 2_000, iq_mode='I+Q')
    ds = do1D(rf_frequency, 80e6, 120e6, 21, 0.0, fast_scan_param, name='frequency_search', reset_param=True).run()

    return ds


def test_ampl():
    pulse = context.init_pulselib(n_gates=2, n_sensors=2, rf_sources=True)

    rf_amplitude = pulse.rf_params['SD2'].source_amplitude
    # NOTE:  reload_seq=True !!!
    fast_scan_param = fast_scan1D_param(pulse, 'P1', 50.0, 21, 2_000, iq_mode='I+Q', reload_seq=True)
    ds = do1D(rf_amplitude, 20.0, 200.0, 10, 0.0, fast_scan_param, name='amplitude_sweep', reset_param=True).run()

    return ds


#%%
if __name__ == '__main__':
    context.init_coretools()
    ds1 = test_freq()
    ds2 = test_ampl()
