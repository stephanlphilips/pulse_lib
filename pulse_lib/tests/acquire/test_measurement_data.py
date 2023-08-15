
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
    context.add_hw_schedule(sequence)
    sequence.upload()
    sequence.play()
    data = sequence.get_measurement_results()
    print(data)

    return None

def test2(iq_mode):
    pulse = context.init_pulselib(n_gates=2, n_sensors=2, rf_sources=True)

    s = pulse.mk_segment()

    s.P1.add_block(0, 20, 100)
    s.P2.add_block(0, 20, 100)

    s.SD1.acquire(0, 1000, wait=True)
    s.SD2.acquire(0, 1000, wait=True)

    sequence = pulse.mk_sequence([s])
    sequence.n_rep = 4
    context.add_hw_schedule(sequence)

    sequence.upload()
    sequence.play()
    data = sequence.get_measurement_results(iq_mode=iq_mode)
    print(data)

    return None


if __name__ == '__main__':
    ds1 = test1()
    ds2 = []
    for iq_mode in ['Complex', 'I', 'Q', 'amplitude', 'phase', 'I+Q', 'amplitude+phase']:
        ds2.append(test2(iq_mode))
