
from pulse_lib.tests.configurations.test_configuration import context

def test1():
    pulse = context.init_pulselib(n_gates=2, n_sensors=2)

    s = pulse.mk_segment()

    s.P1.add_block(0, 20, 100)
    s.P2.add_block(0, 20, -100)
    s.SD1.acquire(0, 10000)
    s.SD2.acquire(0, 10000)
    s.wait(10000)

    sequence = pulse.mk_sequence([s])
    sequence.n_rep = None
    sequence.set_acquisition(sample_rate=500e3)
    m_param = sequence.get_measurement_param()
    context.add_hw_schedule(sequence)
    context.plot_awgs(sequence, print_acquisitions=True)

    return context.run('down-sampling1', sequence, m_param)

def test2():
    pulse = context.init_pulselib(n_gates=2, n_sensors=2)
#    # now with other delays
#    pulse.add_channel_delay('P1', -20)
#    pulse.add_channel_delay('M1', 40)
#    pulse.add_channel_delay('SD1', 20)
#    pulse.add_channel_delay('SD2', 20)

    s = pulse.mk_segment()

    # 2 acquisitions on channel 2
    s.P1.add_block(0, 20, 100)

    s.SD2.acquire(0, 10000)
    s.wait(10000, reset_time=True)
    s.SD2.acquire(0, 10000)
    s.wait(10000)

    sequence = pulse.mk_sequence([s])
    sequence.n_rep = None
    sequence.set_acquisition(sample_rate=500e3)
    m_param = sequence.get_measurement_param()
    context.add_hw_schedule(sequence)
    context.plot_awgs(sequence, print_acquisitions=True)

    return context.run('down-sampling2', sequence, m_param)

if __name__ == '__main__':
    ds1 = test1()
    ds2 = test2()
