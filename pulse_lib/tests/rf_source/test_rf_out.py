
from pulse_lib.tests.configurations.test_configuration import context


#%%
import pulse_lib.segments.utility.looping as lp

def test1():
    pulse = context.init_pulselib(n_gates=2, n_sensors=2, rf_sources=True)

    s = pulse.mk_segment()

    s.P1.add_block(0, 20, 100)
    s.P2.add_block(0, 20, -100)
    s.SD1.acquire(0, 1000, wait=True)
    s.SD2.acquire(0, 1000, wait=True)

    sequence = pulse.mk_sequence([s])
    sequence.n_rep = None
    m_param = sequence.get_measurement_param()

    context.plot_awgs(sequence)

    return context.run('m_param1', sequence, m_param)


def test2():
    pulse = context.init_pulselib(n_gates=2, n_sensors=2, rf_sources=True)

    t_wait = lp.linspace(200, 1000, 5, 't_wait', unit='ns', axis=0)

    s = pulse.mk_segment()

    # 2 acquisitions on channel 2
    s.P1.add_block(0, 20, 100)
    s.wait(480, reset_time=True)

    s.SD2.acquire(0, 1000, wait=True)
    s.wait(t_wait, reset_time=True)
    s.SD2.acquire(0, 1000, wait=True)
    s.wait(1000, reset_time=True)

    sequence = pulse.mk_sequence([s])
    sequence.n_rep = 4
    m_param = sequence.get_measurement_param()

    for i, t in enumerate(t_wait):
        context.plot_awgs(sequence, index=(i,))

    return context.run('m_param2', sequence, m_param)


#%%

if __name__ == '__main__':
    ds1 = test1()
    ds2 = test2()
