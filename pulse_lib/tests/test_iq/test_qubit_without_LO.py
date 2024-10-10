from pulse_lib.tests.configurations.test_configuration import context


#%%

def test1():
    '''
    Test phase shift and MW pulse combinations
    '''
    pulse = context.init_pulselib(n_qubits=2, no_IQ=True)

    s = pulse.mk_segment()

    s.q1.add_MW_pulse(10, 110, 50, 150e6)

    s.q2.add_MW_pulse(200, 300, 80, 200e6)
    s.wait(400)

    context.plot_segments([s])

    sequence = pulse.mk_sequence([s])

    context.plot_awgs(sequence, ylim=(-0.100,0.100))

    return None


if __name__ == '__main__':
    ds1 = test1()
