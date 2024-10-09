
from pulse_lib.tests.configurations.test_configuration import context

#%%
from numpy import pi


def test1():
    pulse = context.init_pulselib(n_gates=1, n_qubits=1)

    f_q1 = pulse.qubit_channels['q1'].resonance_frequency

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


def test2():
    # Off resonant pulses
    pulse = context.init_pulselib(n_gates=1, n_qubits=1)

    f_q1 = pulse.qubit_channels['q1'].resonance_frequency

    s = pulse.mk_segment()

    s.q1.add_MW_pulse(0, 20, 100, f_q1)
    s.reset_time()
    s.q1.add_MW_pulse(0, 20, 100, f_q1-25e6)
    s.reset_time()
    s.q1.add_MW_pulse(0, 20, 100, f_q1)
    s.reset_time()
    s.wait(100, reset_time=True)
    s.q1.add_MW_pulse(0, 12, 100, f_q1)
    s.reset_time()
    s.q1.add_MW_pulse(0, 12, 100, f_q1-25e6)
    s.reset_time()
    s.q1.add_MW_pulse(0, 12, 100, f_q1)

    sequence = pulse.mk_sequence([s])
    sequence.n_rep = None
    context.plot_awgs(sequence)

    return None


def test3():
    # Unaligned pulses
    pulse = context.init_pulselib(n_gates=1, n_qubits=2)

    f_q1 = pulse.qubit_channels['q1'].resonance_frequency
    pulse.qubit_channels['q2'].resonance_frequency = f_q1
    s = pulse.mk_segment()

    s.q1.add_MW_pulse(0, 20, 100, f_q1)
    s.q2.add_MW_pulse(2, 18, 100, f_q1)
    s.reset_time()
    s.q1.add_MW_pulse(0, 20, 100, f_q1)
    s.q2.add_MW_pulse(3, 13, 100, f_q1-25e6)
    s.reset_time()
    s.q1.add_MW_pulse(0, 20, 100, f_q1)
    s.q2.add_MW_pulse(2, 18, 100, f_q1)
    s.reset_time()

    sequence = pulse.mk_sequence([s])
    sequence.n_rep = None
    context.plot_awgs(sequence)

    return None

def test4():
    # Long (unaligned) pulses. Long is > 200 ns.
    pulse = context.init_pulselib(n_gates=1, n_qubits=2)

    f_q1 = pulse.qubit_channels['q1'].resonance_frequency
    pulse.qubit_channels['q2'].resonance_frequency = f_q1
    s = pulse.mk_segment()

    s.q1.add_MW_pulse(2, 100, 100, f_q1)
    s.q1.add_MW_pulse(100, 220, 100, f_q1)
    s.q2.add_MW_pulse(0, 218, 100, f_q1)
    s.reset_time()
    s.q1.add_MW_pulse(0, 220, 100, f_q1)
    s.q2.add_MW_pulse(3, 213, 100, f_q1-25e6)
    s.reset_time()
    s.q1.add_MW_pulse(0, 220, 100, f_q1)
    s.q2.add_MW_pulse(2, 218, 100, f_q1)
    s.reset_time()

    sequence = pulse.mk_sequence([s])
    sequence.n_rep = None
    context.plot_awgs(sequence)

    return None

#%%

if __name__ == '__main__':
    ds1 = test1()
    ds2 = test2()
    ds3 = test3()
    ds4 = test4()
