
from pulse_lib.tests.configurations.test_configuration import context

#%%
from numpy import pi


def test1():
    pulse = context.init_pulselib(n_gates=1, n_qubits=1, drive_with_plungers=True)

    f_q1 = 50e6

    s = pulse.mk_segment()

    s.P1.add_block(0, 100, 200)
    s.q1.add_MW_pulse(0, 20, 100, f_q1)
    s.q1.add_phase_shift(20, pi/2)
    s.q1.add_phase_shift(20, pi/2)
    s.q1.add_MW_pulse(20, 40, 100, f_q1)
    s.q1.add_phase_shift(40, pi)
    s.reset_time()
    s.q1.add_phase_shift(0, -pi/2)
    s.q1.add_MW_pulse(0, 20, 100, f_q1)
    s.q1.add_phase_shift(20, pi/2)
    s.q1.add_MW_pulse(20, 40, 100, f_q1)

    sequence = pulse.mk_sequence([s])
    sequence.n_rep = 1
    context.plot_awgs(sequence)

    return None


#%%

if __name__ == '__main__':
    ds1 = test1()
