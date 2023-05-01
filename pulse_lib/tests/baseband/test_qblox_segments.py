
from pulse_lib.tests.configurations.test_configuration import context

#%%
def test1():
    pulse = context.init_pulselib(n_gates=2)

    segments = []

    s = pulse.mk_segment()
    segments.append(s)

    s.P1.add_ramp_ss(20, 40, 100, 200)

    s = pulse.mk_segment()
    segments.append(s)

    s.P1.add_ramp_ss(20, 40, 200, 0)

    s = pulse.mk_segment()
    segments.append(s)

    s.P1.add_block(20, 40, 150)

    s = pulse.mk_segment()
    segments.append(s)

    s.P1.add_block(20, 40, 150)
    s.wait(100)

    sequence = pulse.mk_sequence(segments)
    sequence.n_rep = 2

    context.plot_awgs(sequence, ylim=(-0.2, 0.20))

#    return context.run('1ns', sequence, m_param)


#%%
if __name__ == '__main__':
    ds1 = test1()

#%%
if False:
    from pulse_lib.tests.utils.last_upload import get_last_upload

    lu = get_last_upload(context.pulse)