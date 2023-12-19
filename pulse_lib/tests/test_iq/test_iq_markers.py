
from pulse_lib.tests.configurations.test_configuration import context
import pulse_lib.segments.utility.looping as lp


#%%
def test1():
    pulse = context.init_pulselib(n_qubits=4, n_sensors=1)
    # f LO: 2.400 and 2.800 GHz
    # qubit freqs: 2.450, 2.550, 2.650, 2.750 GHz
    f_q1 = 2.450e9 # IQ = +50 MHz
    f_q2 = 2.550e9 # IQ = +150 MHz
    f_q3 = 2.650e9 # IQ = -150 MHz
    f_q4 = 2.750e9 # IQ = -50 MHz

    # marker setup + hold = 120 ns. Minimum off time: 20 ns.
    t_wait = lp.linspace(120, 160, 9, name='t_wait', unit='ns', axis=0)
    s = pulse.mk_segment()

    s.q1.add_MW_pulse(0, 100, 200.0, f_q1)
    s.wait(t_wait, reset_time=True)
    s.q2.add_MW_pulse(0, 100, 200.0, f_q2)
    s.wait(t_wait, reset_time=True)
    s.q3.add_MW_pulse(0, 100, 200.0, f_q3)
    s.wait(t_wait, reset_time=True)
    s.q4.add_MW_pulse(0, 100, 200.0, f_q4)

    s.SD1.acquire(0, 100)

    sequence = pulse.mk_sequence([s])
    sequence.n_rep = 10
    m_param = sequence.get_measurement_param()

    for i in range(len(t_wait)):
        context.plot_awgs(sequence, index=(i,))

    return context.run('iq-markers', sequence, m_param)


def test2():
    '''
    Test rounding of marker times.
    Marker times should be rounded to multiples of 4 or 5 ns (depending on hardware)
    There should not be an accumulation of rounding errors.

    Use marker setup_ns=0, hold_ns=0 to check results.
    Minimum off time is 20 ns.
    '''
    pulse = context.init_pulselib(n_qubits=1, n_sensors=1)
    iq_marker = pulse.marker_channels['M_IQ1']
    iq_marker.setup_ns = 0
    iq_marker.hold_ns = 0
#    iq_marker.delay = -10
    # f LO: 2.400 and 2.800 GHz
    # qubit freqs: 2.450, 2.550, 2.650, 2.750 GHz
    f_q1 = 2.450e9 # IQ = +50 MHz

    # marker setup + hold = 120 ns.
    t_wait = lp.linspace(140, 150, 11, name='t_wait', unit='ns', axis=0)
    s = pulse.mk_segment()

    for _ in range(10):
        s.q1.add_MW_pulse(0, 100, 200.0, f_q1)
        s.wait(t_wait, reset_time=True)

    s.wait(200)
    s.SD1.acquire(0, 100)

    sequence = pulse.mk_sequence([s])
    sequence.n_rep = 1
    m_param = sequence.get_measurement_param()

    for i in range(len(t_wait)):
        context.plot_awgs(sequence, index=(i,))

    return context.run('iq-markers-2', sequence, m_param)


#%%

if __name__ == '__main__':
    ds = test1()
    ds = test2()
