
from pulse_lib.tests.configurations.test_configuration import context


#%%
from pulse_lib.segments.conditional_segment import conditional_segment
from pulse_lib.segments.utility.measurement_ref import MeasurementRef

drive_with_plungers = False

def get_feedback_latency(backend):
    if backend in ['Tektronix_5014', 'Keysight']:
        print(f'feedback not supported for {backend}')
        return None
    elif backend == 'Keysight_QS':
        feedback_latency = 700
    else:
        feedback_latency = 500
    return feedback_latency

def test1():
    pulse = context.init_pulselib(n_gates=2, n_qubits=2, n_sensors=2, drive_with_plungers=drive_with_plungers)
    f_q1 = pulse.qubit_channels['q1'].resonance_frequency

    feedback_latency = get_feedback_latency(pulse._backend)
    if feedback_latency is None:
        return

    t_measure = 500

    s = pulse.mk_segment('seg1')
    s1 = s

    s.P1.add_block(10, 20, -10)
    s.wait(30, reset_time=True)
    s.SD1.acquire(0, t_measure, 'm1', threshold=0.0015, zero_on_high=True, wait=True)
    s.wait(feedback_latency, reset_time=True)

    s_true = pulse.mk_segment()
    s_false = pulse.mk_segment()

    s_false.q1.add_MW_pulse(10, 20, 80.0, f_q1)
    s_false.wait(10, reset_time=True)

    cond_seg1 = conditional_segment(MeasurementRef('m1'), [s_false, s_true], name='cond')

    sequence = pulse.mk_sequence([s1, cond_seg1])
    sequence.n_rep = 3

    context.plot_awgs(sequence, ylim=(-0.100,0.100))

    m_param = sequence.get_measurement_param()
    return context.run('feedback', sequence, m_param)


def test2():
    pulse = context.init_pulselib(n_gates=2, n_qubits=2, n_sensors=2)
    f_q1 = pulse.qubit_channels['q1'].resonance_frequency

    feedback_latency = get_feedback_latency(pulse._backend)
    if feedback_latency is None:
        return

    t_measure = 500

    s = pulse.mk_segment('seg1')
    s1 = s

    s.P1.add_block(10, 20, -10)
    s.wait(30, reset_time=True)
    s.SD1.acquire(0, t_measure, 'm0', threshold=0.0024, zero_on_high=False, wait=True)
    s.wait(200, reset_time=True)
    s.SD1.acquire(0, t_measure, 'm1', threshold=0.0024, zero_on_high=False, wait=True)
    s.wait(feedback_latency, reset_time=True)

    s_true = pulse.mk_segment()
    s_false = pulse.mk_segment()

    s_false.q1.add_MW_pulse(10, 20, 80.0, f_q1)
    s_false.wait(10, reset_time=True)

    cond_seg1 = conditional_segment(MeasurementRef('m0'), [s_false, s_true], name='cond')

    sequence = pulse.mk_sequence([s1, cond_seg1])
    sequence.n_rep = 3

    context.plot_awgs(sequence, ylim=(-0.100,0.100))

    m_param = sequence.get_measurement_param()
    return context.run('feedback', sequence, m_param)


def test3():
    pulse = context.init_pulselib(n_gates=2, n_qubits=2, n_sensors=2)
    f_q1 = pulse.qubit_channels['q1'].resonance_frequency
    f_q2 = pulse.qubit_channels['q2'].resonance_frequency

    feedback_latency = get_feedback_latency(pulse._backend)
    if feedback_latency is None:
        return

    t_measure = 500

    s = pulse.mk_segment('seg1')
    s1 = s

    s.P1.add_block(10, 20, -10)
    s.wait(30, reset_time=True)
    s.SD1.acquire(0, t_measure, 'm0', threshold=0.0024, zero_on_high=False, wait=True)
    s.wait(200, reset_time=True)
    s.SD2.acquire(0, t_measure, 'm1', threshold=0.0024, zero_on_high=False, wait=True) # @@@ SD1 => Weird error
    s.wait(feedback_latency, reset_time=True)

    s_true = pulse.mk_segment()
    s_false = pulse.mk_segment()

    s_false.q1.add_MW_pulse(10, 20, 80.0, f_q1)
    s_false.wait(40, reset_time=True)

    cond_seg1 = conditional_segment(MeasurementRef('m0'), [s_false, s_true], name='cond')

    s_true = pulse.mk_segment()
    s_false = pulse.mk_segment()

    s_false.q2.add_MW_pulse(10, 20, 80.0, f_q2)
    s_false.wait(20, reset_time=True)

    cond_seg2 = conditional_segment(MeasurementRef('m1'), [s_false, s_true], name='cond')

    s_end = pulse.mk_segment()
    s_end.wait(100)

    sequence = pulse.mk_sequence([s1, cond_seg1, cond_seg2, s_end])
    sequence.n_rep = 3

    context.plot_awgs(sequence, ylim=(-0.100,0.100))

    m_param = sequence.get_measurement_param()
    return context.run('feedback', sequence, m_param)


#%%
if __name__ == '__main__':
    context.init_coretools()
    ds1 = test1()
    ds2 = test2()
    print(ds2.m1_1(), ds2.m1_2(), ds2.m1_3(), ds2.m1_4())

    ds3 = test3()
