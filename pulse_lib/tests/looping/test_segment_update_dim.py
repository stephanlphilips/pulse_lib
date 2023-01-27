
from pulse_lib.tests.configurations.test_configuration import context
import pulse_lib.segments.utility.looping as lp

def test():
    pulse = context.init_pulselib(n_gates=2)

    n_pulses = lp.array([1,2,4,9], axis=0, name='n_pulses')

    s = pulse.mk_segment()

    context.segment = s

    s.P1.update_dim(n_pulses)
    for i,n in enumerate(n_pulses):
        p1 = s.P1[i]
        for _ in range(int(n)):
            p1.add_ramp_ss(0, 100, -80, 80)
            p1.reset_time()

    s.P2.add_block(0, 100, 60)

    for i in range(len(n_pulses)):
        context.plot_segments([s], index=[i])

    sequence = pulse.mk_sequence([s])
    context.add_hw_schedule(sequence)
    for n in sequence.n_pulses.values:
        sequence.n_pulses(n)
        context.plot_awgs(sequence, ylim=(-0.100,0.100))

    return None

if __name__ == '__main__':
    ds = test()
