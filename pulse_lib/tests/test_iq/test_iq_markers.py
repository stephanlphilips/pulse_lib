
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

    # marker setup + hold = 120 ns.
    t_wait = lp.linspace(110, 150, 9, name='t_wait', unit='ns', axis=0)
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

    context.add_hw_schedule(sequence)
    for i in range(len(t_wait)):
        context.plot_awgs(sequence, index=(i,))

    return context.run('iq-markers', sequence, m_param)

#%%

if __name__ == '__main__':
    ds = test1()
