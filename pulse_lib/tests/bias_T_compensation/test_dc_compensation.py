
from pulse_lib.tests.configurations.test_configuration import context

'''
Test DC compensation pulses after AWG sequence.
NOTE:
    context sets: pulse.add_channel_compensation_limit(gate, (-100, 50))
'''
#%%


def test1():
    pulse = context.init_pulselib(n_gates=4)

    s = pulse.mk_segment()

    s.P1.add_block(0, 200, 10)
    s.P2.add_block(0, 40_000, -100)
    s.P3.add_block(0, 40_000, -200)
    s.P4.add_block(0, 10_000, 10)

    context.plot_segments([s])

    sequence = pulse.mk_sequence([s])
    sequence.n_rep = 2
    context.plot_awgs(sequence, print_acquisitions=True)

    return None


def test2():
    pulse = context.init_pulselib(n_gates=4)
    pulse.add_channel_attenuation('P1', 0.1)
    pulse.add_channel_attenuation('P3', 0.1)

    s = pulse.mk_segment()

    s.P1.add_block(0, 200, 10)
    s.P2.add_block(0, 40_000, -100)
    s.P3.add_block(0, 40_000, -40)
    s.P4.add_block(0, 10_000, 10)

    context.plot_segments([s])

    sequence = pulse.mk_sequence([s])
    sequence.n_rep = 2
    context.plot_awgs(sequence, print_acquisitions=True)

    return None


if __name__ == '__main__':
    ds = test1()
    ds = test2()
