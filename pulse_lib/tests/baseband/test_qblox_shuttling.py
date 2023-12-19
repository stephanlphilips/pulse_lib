
from pulse_lib.tests.configurations.test_configuration import context

#%%


def test1(t):
    pulse = context.init_pulselib(n_gates=1, n_markers=1, n_sensors=1)

    segments = []

    s = pulse.mk_segment()
    segments.append(s)

    n = 20

    s.P1.add_block(0, 20, -100)
    s.M1.add_marker(0, 20)
    s.reset_time()

    for i in range(n):
        s.P1.add_ramp_ss(0, t, -100, 100)
        s.reset_time()
        s.P1.add_block(0, t, 100)
        s.reset_time()
        s.P1.add_ramp_ss(0, t, 100, -100)
        s.reset_time()
        s.P1.add_block(0, t, -100)
        s.reset_time()

    s.reset_time()
    s.SD1.acquire(0, 20)
    s.wait(20)

    sequence = pulse.mk_sequence(segments)
    sequence.n_rep = 2

    context.plot_awgs(sequence, ylim=(-0.2, 0.2))

    m_param = sequence.get_measurement_param()

    return context.run('shuttle', sequence, m_param)


def test2(t):
    pulse = context.init_pulselib(n_gates=1, n_markers=1, n_sensors=1)
    if pulse._backend in ['Qblox']:
        from pulse_lib.qblox.pulsar_sequencers import PulsarConfig
        PulsarConfig.NS_SUB_DIVISION = 10

    segments = []

    s = pulse.mk_segment(hres=True)
    segments.append(s)

    n = 1000

    s.P1.add_block(0, 20, -100)
    s.M1.add_marker(0, 20)
    s.reset_time()

    for i in range(n):
        s.P1.add_ramp_ss(0, t, -100, 100)
        s.reset_time()
        s.P1.add_block(0, t, 100)
        s.reset_time()
        s.P1.add_ramp_ss(0, t, 100, -100)
        s.reset_time()
        s.P1.add_block(0, t, -100)
        s.reset_time()

    s.reset_time()
    s.SD1.acquire(0, 20)
    s.wait(20)

    sequence = pulse.mk_sequence(segments)
    sequence.n_rep = 2

    context.plot_awgs(sequence, ylim=(-0.2, 0.2))

    m_param = sequence.get_measurement_param()

    return context.run('shuttle', sequence, m_param)


#%%
if __name__ == '__main__':
    ds1 = test1(121) # not aligned
    ds1 = test1(60) # aligned
    ds1 = test1(40) # aligned
    ds1 = test1(41) # not aligned
    ds1 = test1(20) # shorter
    ds1 = test1(17)
    ds1 = test1(8)
    ds1 = test1(7)
    ds1 = test1(4)
    ds1 = test1(3)
    ds1 = test1(2)
    ds1 = test1(1)
    # hres
    ds1 = test2(4.1)
    ds1 = test2(9.34)


#%%
if False:
    from pulse_lib.tests.utils.last_upload import get_last_upload

    lu = get_last_upload(context.pulse)