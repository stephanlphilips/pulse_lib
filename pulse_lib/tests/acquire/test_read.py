
from pulse_lib.tests.configurations.test_configuration import context
from pulse_lib.scan.read_input import read_channels

#%%
def test1():
    pulse = context.init_pulselib(n_gates=1, n_sensors=2, rf_sources=False)

    dc_param = read_channels(pulse, 1000)

    return context.run('read', dc_param)


def test2():
    pulse = context.init_pulselib(n_gates=1, n_sensors=2, rf_sources=False)

    dc_param = read_channels(pulse, 100000, sample_rate=500e3)

    return context.run('read', dc_param)


def test3(iq_mode='I+Q'):
    pulse = context.init_pulselib(n_gates=1, n_sensors=2, rf_sources=True)

    dc_param = read_channels(pulse, 10000, iq_mode=iq_mode)

    return context.run('read_iq_'+iq_mode, dc_param)


def test4():
    # takes 100 seconds to run !!!
    pulse = context.init_pulselib(n_gates=1, n_sensors=2, rf_sources=False)
    if pulse._backend in ['Keysight', 'Keysight_QS']:
        for awg in pulse.awg_devices.values():
            # 1e8 samples at 1e6 Sa/s => 100 sec.
            awg.set_waveform_limit(1e8)

    dc_param = read_channels(pulse, 100e9, sample_rate=100) # 100 Hz, 100 seconds

    return context.run('read', dc_param)


#%%
if __name__ == '__main__':
    ds1 = test1()
    ds2 = test2()
    ds3 = test3()
    ds4 = test4()
