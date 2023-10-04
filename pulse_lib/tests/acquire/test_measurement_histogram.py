
from pulse_lib.tests.configurations.test_configuration import context


# %%
def test1():
    pulse = context.init_pulselib(n_gates=2, n_sensors=2)

    s = pulse.mk_segment()

    s.P1.add_block(0, 20, 100)
    s.P2.add_block(0, 20, -100)
    s.SD1.acquire(0, 1000, 'm1', wait=True)
    s.SD2.acquire(0, 1000, wait=True, threshold=3000, accept_if=1)

    sequence = pulse.mk_sequence([s])
    sequence.n_rep = 10
    m_param = sequence.get_measurement_param(accept_mask=True)

    m_param.add_measurement_histogram('m1', 20, (0.0, 20_000.0))  # (m_name, bins, (min,nax))
    m_param.add_sensor_histogram('SD1', 20, (0.0, 20_000.0))
    m_param.add_sensor_histogram('SD2', 20, (0.0, 20_000.0), accepted_only=True)

    context.add_hw_schedule(sequence)

    return context.run('histogram', sequence, m_param)


# %%
if __name__ == '__main__':
    ds1 = test1()
