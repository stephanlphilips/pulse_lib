
from pulse_lib.tests.configurations.test_configuration import context

#%%
import pulse_lib.segments.utility.looping as lp


def test():
    pulse = context.init_pulselib(n_gates=2)

    n_pulses = lp.linspace(1,4,4, axis=1, name='n_pulses')
    t_wait = lp.linspace(10,20,3, axis=0, name='t_wait')

    s = pulse.mk_segment()

    context.segment = s

#    s.update_dim(t)
#    s.wait(0*t_wait)
    s.update_dim(n_pulses)
    for i,n in enumerate(n_pulses):
        p1 = s[i].P1
        for _ in range(int(n)):
#            p1.wait(t_wait)
            p1.reset_time()
            p1.add_ramp_ss(0, 20, -80, 80)
            p1.reset_time()

    s.P2.wait(t_wait)
    s.P2.reset_time()
    s.P2.add_block(0, 100, 60)

#    for i in range(len(n_pulses)):
#        context.plot_segments([s], index=[i,1])

    sequence = pulse.mk_sequence([s])
    sequence.n_rep = 2
    for n in sequence.n_pulses.values:
        for t in sequence.t_wait.values:
            sequence.n_pulses(n)
            sequence.t_wait(t)
            context.plot_awgs(sequence, ylim=(-0.100,0.100))

    return None


if __name__ == '__main__':
    ds = test()
