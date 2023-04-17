
from pulse_lib.tests.configurations.test_configuration import context

from numpy import pi

#%%
def test1():
    pulse = context.init_pulselib(n_qubits=1)
    pulse.set_qubit_correction_phase('q1', 0.1*pi)

    s = pulse.mk_segment()

    s.q1.add_MW_pulse(0, 20, 100, 2.450e9)

    context.plot_segments([s])

    sequence = pulse.mk_sequence([s])
    sequence.n_rep = 1
    context.add_hw_schedule(sequence)
    context.plot_awgs(sequence)

    return None

def test2():
    pulse = context.init_pulselib(n_qubits=1)
    pulse.set_qubit_correction_gain('q1', 1.0, 1.1)

    s = pulse.mk_segment()

    s.q1.add_MW_pulse(0, 20, 100, 2.450e9)

    context.plot_segments([s])

    sequence = pulse.mk_sequence([s])
    sequence.n_rep = 1
    context.add_hw_schedule(sequence)
    context.plot_awgs(sequence)

    return None

if __name__ == '__main__':
    ds1 = test1()
    ds2 = test2()
