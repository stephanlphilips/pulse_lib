
from pulse_lib.tests.configurations.test_configuration import context

from numpy import pi

def test():
    pulse = context.init_pulselib(n_qubits=1)

    s = pulse.mk_segment()

    s.q1.add_MW_pulse(0, 20, 100, 2.450e9)
    s.q1.add_phase_shift(20, pi/2)
    s.q1.add_phase_shift(20, pi/2)
    s.q1.add_MW_pulse(20, 40, 100, 2.450e9)
    s.q1.add_phase_shift(40, pi)
    s.reset_time()
    s.q1.add_phase_shift(0, -pi/2)
    s.q1.add_MW_pulse(0, 20, 100, 2.450e9)
    s.q1.add_phase_shift(20, pi/2)
    s.q1.add_MW_pulse(20, 40, 100, 2.450e9)

    context.plot_segments([s])

    sequence = pulse.mk_sequence([s])
    sequence.n_rep = 1
    context.add_hw_schedule(sequence)
    context.plot_awgs(sequence)

    return None

if __name__ == '__main__':
    ds = test()
