
from pulse_lib.tests.configurations.test_configuration import context


#%%
def test():
    pulse = context.init_pulselib(n_gates=2, n_qubits=3, n_sensors=2, n_markers=1)

    s = pulse.mk_segment()

    s.P1.wait(100)
    s.P1.add_block(0, 20, 100)
    s.P2.add_block(0, 20, -100)
    s.M1.add_marker(0, 20)
    s.q1.add_MW_pulse(0, 20, 100, 2.450e9)
    s.q3.add_MW_pulse(0, 20, 100, 2.650e9)
    s.SD1.acquire(0, 100)
    s.SD2.acquire(0, 100)

#    context.plot_segments([s])

    sequence = pulse.mk_sequence([s])
    sequence.n_rep = 1
    context.plot_awgs(sequence, print_acquisitions=True)

    # now with other delays
    pulse.add_channel_delay('P1', -20)
    pulse.add_channel_delay('M1', 40)
    pulse.add_channel_delay('SD1', 20)
    pulse.add_channel_delay('SD2', 20)

    sequence = pulse.mk_sequence([s])
    sequence.n_rep = 1
    context.plot_awgs(sequence, print_acquisitions=True)

    return None

def test2():
    pulse = context.init_pulselib(n_gates=2, n_sensors=2,
                                  rf_sources=True)

    s = pulse.mk_segment()

    s.P1.wait(100)
    s.P1.add_block(0, 20, 100)
    s.P2.add_block(0, 20, -100)
    s.SD1.acquire(0, 100)
    s.SD2.acquire(0, 100)

    # now with other delays
    pulse.add_channel_delay('P1', 0)
    pulse.add_channel_delay('SD1', 0)
    pulse.add_channel_delay('SD2', 0)
    pulse.digitizer_channels['SD2'].rf_source.mode = 'continuous'
    pulse.digitizer_channels['SD2'].rf_source.startup_time_ns = 0
    pulse.digitizer_channels['SD2'].rf_source.prolongation_ns = 0

    sequence = pulse.mk_sequence([s])
    sequence.n_rep = 10
    context.plot_awgs(sequence, print_acquisitions=True)

    return None

#%%


if __name__ == '__main__':
    ds = test()
    ds2 = test2()
