
from pulse_lib.tests.configurations.test_configuration import context

def test():
    pulse = context.init_pulselib(n_qubits=2)

    s = pulse.mk_segment()

    s.q1.add_phase_shift(0, -0.1)
    s.q1.add_phase_shift(5, 0.1)
    s.q1.add_phase_shift(6, 0.1)
    s.q1.add_MW_pulse(10, 21, 50, 2.45e9)
    s.q1.add_phase_shift(21, 0.2)
    s.q1.add_phase_shift(24, 0.05)

    s.q1.add_phase_shift(26, 0.3)

    s.q1.add_phase_shift(30, 0.4)
    s.q1.add_MW_pulse(31, 40, 200, 2.45e9)
    s.q1.add_phase_shift(40, 0.5)

    context.plot_segments([s])

    sequence = pulse.mk_sequence([s])
    context.add_hw_schedule(sequence)
    context.plot_awgs(sequence, ylim=(-0.100,0.100))
    context.pulse.awg_devices['AWG3'].describe()

    return None

if __name__ == '__main__':
    ds = test()
