
from pulse_lib.tests.configurations.test_configuration import context
from pulse_lib.scan.read_input import read_channels

#%%
def test1():
    pulse = context.init_pulselib(n_gates=0, n_sensors=2, rf_sources=False)

    dc_param = read_channels(pulse, 1000)

    return context.run('read', dc_param)

def test2():
    pulse = context.init_pulselib(n_gates=0, n_sensors=2, rf_sources=False)

    dc_param = read_channels(pulse, 100000, sample_rate=500e3)

    return context.run('read', dc_param)

def test3(iq_mode='I+Q'):
    pulse = context.init_pulselib(n_gates=0, n_sensors=2, rf_sources=True)

    dc_param = read_channels(pulse, 10000, iq_mode=iq_mode)

    return context.run('read_iq_'+iq_mode, dc_param)

#%%
if __name__ == '__main__':
    ds1 = test1()
    ds2 = test2()
    ds3 = test3()
