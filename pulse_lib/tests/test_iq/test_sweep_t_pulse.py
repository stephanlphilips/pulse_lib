from pulse_lib.tests.configurations.test_configuration import context

#%%
import numpy as np


def test1(t_pulse_min=1):
    '''
    Vary pulse length and offset.
    Test is relevant for Qblox and Keysight_QS uploaders.
    '''
    pulse = context.init_pulselib(n_qubits=2)

    s = pulse.mk_segment()

    for offset in np.arange(0, 10):
        for t_pulse in np.arange(0, 10):
            s.wait(100+offset, reset_time=True)
            s.q1.add_MW_pulse(0, t_pulse+t_pulse_min, 50, 2.45e9)

    context.plot_segments([s])

    sequence = pulse.mk_sequence([s])
    context.plot_awgs(sequence)

    return None


if __name__ == '__main__':
    ds1 = test1()
    ds2 = test1(200)
