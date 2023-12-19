
from pulse_lib.tests.configurations.test_configuration import context


#%%
def test1():
    pulse = context.init_pulselib(n_gates=2, n_sensors=2)

    s = pulse.mk_segment()

    s.P1.add_block(0, 20, 100)
    s.P2.add_block(0, 20, -100)
    s.SD1.acquire(0, 1000, wait=True)
    s.SD2.acquire(0, 1000, wait=True)

    sequence = pulse.mk_sequence([s])
    sequence.n_rep = None
    m_param = sequence.get_measurement_param()

    return context.run('m_param1', sequence, m_param)


def test2():
    pulse = context.init_pulselib(n_gates=2, n_sensors=2)

    s = pulse.mk_segment()

    # 2 acquisitions on channel 2
    s.P1.add_block(0, 20, 100)

    s.SD2.acquire(0, 1000, wait=True)
    s.reset_time()
    s.SD2.acquire(0, 1000, wait=True)

    sequence = pulse.mk_sequence([s])
    sequence.n_rep = 4
    m_param = sequence.get_measurement_param()

    return context.run('m_param2', sequence, m_param)


def test3(iq_mode):
    pulse = context.init_pulselib(n_gates=2, n_sensors=2, rf_sources=True)

    s = pulse.mk_segment()

    s.P1.add_block(0, 20, 100)
    s.P2.add_block(0, 20, 100)

    s.SD1.acquire(0, 1000, wait=True)
    s.SD2.acquire(0, 1000, wait=True)

    sequence = pulse.mk_sequence([s])
    sequence.n_rep = 4
    m_param = sequence.get_measurement_param(iq_mode=iq_mode)

    return context.run('m_param3_'+iq_mode, sequence, m_param)


def test4():
    pulse = context.init_pulselib(n_gates=2, n_sensors=2)

    s = pulse.mk_segment()

    s.P1.add_block(0, 20, 100) #lp.linspace(-100, 100, 5, axis=0, name='amplitude', unit='mV'))
    s.P2.add_block(0, 20, -100)
    s.SD1.acquire(0, 1000, threshold=1.0, wait=True)
    s.SD2.acquire(0, 1000, threshold=2.0, wait=True)

    sequence = pulse.mk_sequence([s])
    sequence.n_rep = 4
    m_param = sequence.get_measurement_param()

    return context.run('m_param2', sequence, m_param)


#%%

if __name__ == '__main__':
    ds1 = test1()
    ds2 = test2()
    ds3 = []
    for iq_mode in [#'Complex',
                    'I', 'Q', 'amplitude', 'phase', 'I+Q', 'amplitude+phase']:
        ds3.append(test3(iq_mode))
    ds4 = test4()
