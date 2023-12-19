
from pulse_lib.tests.configurations.test_configuration import context

#%%
import pulse_lib.segments.utility.looping as lp


def test():
    pulse = context.init_pulselib(n_gates=1, n_sensors=1)

    n_pulses = lp.array([1,2,5,10], name='n_pulses', axis=1)
    t_all = lp.linspace(1000, 3000, 3, name='t_all', axis=0)
    t_pulse = t_all / n_pulses

    s = pulse.mk_segment()
    s.update_dim(n_pulses)

    # Add other axis before indexing. It cannot be added on the slice.
    # TODO: This shouldn't be necessary...
    s.update_dim(t_all)

    for i in range(len(n_pulses)):
        s_i = s[i]
        for j in range(int(n_pulses[i])):
            s_i.P1.add_block(0, t_pulse[i], 50)
            s_i.reset_time()
            s_i.P1.add_block(0, t_pulse[i], -50)
            s_i.reset_time()

    s.wait(100)
    s.SD1.acquire(0, 100)

    context.plot_segments([s], index=(0,0))

    sequence = pulse.mk_sequence([s])
    sequence.n_rep = 2
    m_param = sequence.get_measurement_param()
    for t in sequence.t_all.values:
        sequence.t_all(t)
        for i in sequence.n_pulses.values:
            sequence.n_pulses(i)
            context.plot_awgs(sequence, ylim=(-0.100,0.100))

    ds = context.run('test_2D_divide', sequence, m_param)

    return ds


if __name__ == '__main__':
    ds = test()
