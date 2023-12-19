from pulse_lib.tests.configurations.test_configuration import context

#%%
from numpy import pi


def test1():
    '''
    Test phase shift and MW pulse combinations
    '''
    pulse = context.init_pulselib(n_qubits=2)

    s = pulse.mk_segment()

    s.q1.add_phase_shift(0, -0.1*pi)
    s.q1.add_phase_shift(5, 0.1*pi)
    s.q1.add_phase_shift(6, 0.1*pi)
    s.q1.add_MW_pulse(10, 21, 50, 2.45e9)
    s.q1.add_phase_shift(21, 0.2*pi)
    s.q1.add_phase_shift(24, 0.05*pi)

    s.q1.add_phase_shift(26, 0.3*pi)

    s.q1.add_phase_shift(30, 0.4*pi)
    s.q1.add_MW_pulse(31, 40, 80, 2.45e9)
    s.q1.add_phase_shift(40, 0.5*pi)
    s.q1.add_MW_pulse(40, 50, 80, 2.45e9)

    context.plot_segments([s])

    sequence = pulse.mk_sequence([s])
    context.plot_awgs(sequence, ylim=(-0.100,0.100))
    context.pulse.awg_devices['AWG3'].describe()

    return None


def test2():
    '''
    Test long and short waveform rendering.
    '''
    pulse = context.init_pulselib(n_qubits=2)

    s = pulse.mk_segment()

    s.q1.add_MW_pulse(0, 20, 50, 2.45e9)
    s.reset_time()
    s.q1.add_MW_pulse(0, 10, 50, 2.45e9)
    s.reset_time()
    s.q1.add_MW_pulse(0, 50, 50, 2.45e9)
    s.reset_time()
    s.q1.add_MW_pulse(0, 10, 50, 2.45e9)

    context.plot_segments([s])

    sequence = pulse.mk_sequence([s])
    context.plot_awgs(sequence, ylim=(-0.100,0.100))
    context.pulse.awg_devices['AWG3'].describe()

    return None


if __name__ == '__main__':
    ds1 = test1()
    ds2 = test2()
