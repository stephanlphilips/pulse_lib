
from pulse_lib.tests.configurations.test_configuration import context


#%%
import pulse_lib.segments.utility.looping as lp


def test1():
    '''
    Sweep both f_res and f_drive
    '''
    pulse = context.init_pulselib(n_qubits=1, n_sensors=1)
    # f LO: 2.400
    f_q1 = 2.450e9 # IQ = +50 MHz

    f_q1 = lp.linspace(2.410e9, 2.500e9, 10, name='f_res', unit='Hz', axis=0)
    s = pulse.mk_segment()

    s.q1.add_MW_pulse(0, 100, 200.0, f_q1)
    s.wait(20, reset_time=True)
    s.q1.add_MW_pulse(0, 100, 200.0, f_q1)

    s.SD1.acquire(0, 100)

    sequence = pulse.mk_sequence([s])
    sequence.n_rep = 1
    sequence.set_qubit_resonance_frequency('q1', f_q1)
    m_param = sequence.get_measurement_param()

    for i in range(len(f_q1)):
        context.plot_awgs(sequence, index=(i,))

    return context.run('iq-markers', sequence, m_param)


def test2():
    '''
    Sweep only f_res. f_drive is constant.
    Note: currently fails for "Keysight"
    '''
    pulse = context.init_pulselib(n_qubits=1, n_sensors=1)
    # f LO: 2.400
    f_q1_drive = 2.450e9 # IQ = +50 MHz

    f_q1 = lp.linspace(2.410e9, 2.500e9, 10, name='f_res', unit='Hz', axis=0)
    s = pulse.mk_segment()

    s.q1.add_MW_pulse(0, 100, 200.0, f_q1_drive)
    s.wait(20, reset_time=True)
    s.q1.add_MW_pulse(0, 100, 200.0, f_q1_drive)

    s.SD1.acquire(0, 100)

    sequence = pulse.mk_sequence([s])
    sequence.n_rep = 1
    sequence.set_qubit_resonance_frequency('q1', f_q1)
    m_param = sequence.get_measurement_param()

    for i in range(len(f_q1)):
        context.plot_awgs(sequence, index=(i,))

    return context.run('iq-markers', sequence, m_param)


#%%

if __name__ == '__main__':
    ds1 = test1()
    ds2 = test2()
