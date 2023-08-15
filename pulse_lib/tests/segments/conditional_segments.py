
from pulse_lib.tests.configurations.test_configuration import context


#%%
from pulse_lib.segments.conditional_segment import conditional_segment
from pulse_lib.segments.utility.measurement_ref import MeasurementRef

def test1():
    pulse = context.init_pulselib(n_gates=2, n_qubits=2, n_sensors=2)

    t_measure = 500

    s = pulse.mk_segment('seg1')
    s1 = s

    s.P1.add_block(10, 20, -10)
    s.wait(30, reset_time=True)
    s.SD1.acquire(0, t_measure, 'm1', threshold=0.0015, zero_on_high=True, wait=True)
    s.wait(300, reset_time=True)

    s_true = pulse.mk_segment()
    s_false = pulse.mk_segment()

    s_false.q1.add_MW_pulse(10, 20, 80.0, 2.450e9)
    s_false.wait(10, reset_time=True)

    cond_seg1 = conditional_segment(MeasurementRef('m1'), [s_false, s_true], name='cond')

    sequence = pulse.mk_sequence([s1, cond_seg1])
    sequence.n_rep = 3
    context.add_hw_schedule(sequence)

    context.plot_awgs(sequence, ylim=(-0.100,0.100))

    m_param = sequence.get_measurement_param()
    return context.run('feedback', sequence, m_param)


def test2():
    pulse = context.init_pulselib(n_gates=2, n_qubits=2, n_sensors=2)

    t_measure = 500

    s = pulse.mk_segment('seg1')
    s1 = s

    s.P1.add_block(10, 20, -10)
    s.wait(30, reset_time=True)
    s.SD1.acquire(0, t_measure, 'm0', threshold=0.0025, zero_on_high=True, wait=True)
    s.wait(300, reset_time=True)
    s.SD1.acquire(0, t_measure, 'm1', threshold=0.0025, zero_on_high=True, wait=True)
    s.wait(300, reset_time=True)

    s_true = pulse.mk_segment()
    s_false = pulse.mk_segment()

    s_false.q1.add_MW_pulse(10, 20, 80.0, 2.450e9)
    s_false.wait(10, reset_time=True)

    cond_seg1 = conditional_segment(MeasurementRef('m0'), [s_false, s_true], name='cond')

    sequence = pulse.mk_sequence([s1, cond_seg1])
    sequence.n_rep = 3
    context.add_hw_schedule(sequence)

    context.plot_awgs(sequence, ylim=(-0.100,0.100))

    m_param = sequence.get_measurement_param()
    return context.run('feedback', sequence, m_param)


#%%
if __name__ == '__main__':
    context.init_coretools()
#    ds1 = test1()
    ds2 = test2()

