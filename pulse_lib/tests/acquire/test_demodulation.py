
from pulse_lib.tests.configurations.test_configuration import context

#%%
import numpy as np


def test1():
    pulse = context.init_pulselib(n_gates=2, n_sensors=2, rf_sources=True)

    f1 = 10e6
    f2 = 20e6
    pulse.digitizer_channels['SD1'].iq_out = True
    pulse.digitizer_channels['SD1'].frequency = f1
    pulse.digitizer_channels['SD2'].iq_out = True
    pulse.digitizer_channels['SD2'].frequency = f2

    backend = context.pulse._backend
    if backend == 'Tektronix_5014':
        # NOTE: assume M4i. Test does not yet work with other digitizers.
        digitizer = context.station.Dig1
        digitizer.sample_rate(250e6)
        # get actual sample rate from m4i.
        sample_rate = digitizer.sample_rate()
        # WORKAROUND for M4i: multiply amount of data by 2, because there are 2 triggers.
        trigger_factor = 2
    elif backend in ['Keysight', 'Keysight_QS']:
        sample_rate = 100e6
        trigger_factor = 1
    else:
        print(f'Test cannot be executed on {backend}')
        return

    t_measure = 100

    s = pulse.mk_segment()

    s.SD1.acquire(0, t_measure)
    s.wait(10000, reset_time=True)
    s.SD2.acquire(10, t_measure)
    s.wait(10000)

    sequence = pulse.mk_sequence([s])
    sequence.n_rep = 2

    m_param = sequence.get_measurement_param(iq_mode='amplitude+phase')
    context.add_hw_schedule(sequence)

    t = np.arange(0, t_measure, 1e9/sample_rate)  # [ns]
    t *= 1e-9  # [s]

    context.set_mock_data({
            'SD1': [np.cos(2*np.pi*f1*t+np.pi/10)]*trigger_factor,
            'SD2': [0.6*np.exp(1j*(2*np.pi*f2*t+np.pi/5))]*trigger_factor,
            },
            repeat=sequence.n_rep,
            )

    return context.run('demodulation', sequence, m_param)


if __name__ == '__main__':
    ds1 = test1()
    print(ds1.SD1_1_amp)
    print(ds1.SD1_1_phase)
    print(ds1.SD2_1_amp)
    print(ds1.SD2_1_phase)
#    print(ds1.m1_1())
#    print(ds1.m1_2())
#    print(ds1.m1_3())
#    print(ds1.m1_4())
