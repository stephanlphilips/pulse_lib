
from pulse_lib.tests.configurations.test_configuration import context

def test():
    pulse = context.init_pulselib(n_qubits=1)

    s = pulse.mk_segment()

    s.q1.add_chirp(0, 2000, 2.401e9, 2.410e9, 1000)

    context.plot_segments([s])

    sequence = pulse.mk_sequence([s])
    sequence.n_rep = 1
    context.add_hw_schedule(sequence)
    context.plot_awgs(sequence)

    return None

if __name__ == '__main__':
    ds = test()
