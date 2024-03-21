
from pulse_lib.tests.configurations.test_configuration import context

#%%
from pulse_lib.scan.read_input import scan_resonator_frequency


def test1():
    pulse = context.init_pulselib(n_gates=1, n_sensors=2, rf_sources=True)

    scan_param = scan_resonator_frequency(
        pulse, 'SD1', 500, 100e6, 200e6, 1e6, iq_mode='I+Q')

    return context.run('freq_scan', scan_param)


def test2():
    pulse = context.init_pulselib(n_gates=1, n_sensors=2, rf_sources=True)

    scan_param = scan_resonator_frequency(
        pulse, 'SD1', 500, 100e6, 200e6, 30e6,
        n_rep=10,
        iq_mode='I+Q')

    return context.run('freq_scan', scan_param)


def test3():
    pulse = context.init_pulselib(n_gates=1, n_sensors=2, rf_sources=True)

    scan_param = scan_resonator_frequency(
        pulse, 'SD1', 500, 100e6, 200e6, 10e6,
        n_rep=10,
        average_repetitions=True,
        iq_mode='I+Q')

    return context.run('freq_scan', scan_param)


#%%
if __name__ == '__main__':
    ds1 = test1()
    ds2 = test2()
    ds3 = test3()
