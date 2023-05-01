
from pulse_lib.tests.configurations.test_configuration import context
import pulse_lib.segments.utility.looping as lp
from numpy import pi

#%%
def test1():
    pulse = context.init_pulselib(n_gates=2,n_markers=1,n_sensors=1,n_qubits=1)
#    context.station.AWG1.set_digital_filter_mode(3)

    s = pulse.mk_segment()

    s.P1.add_block(110, 111, 1000)
    s.P2.add_block(110, 200, 1000)
    s.q1.add_MW_pulse(110,120,500,2.3e9)#,pi/2)
    s.SD1.acquire(10, 100)
    s.M1.add_marker(10, 120)
    s.wait(90000)

    sequence = pulse.mk_sequence([s])
    sequence.n_rep = 2000
    m_param = sequence.get_measurement_param()
    context.add_hw_schedule(sequence)

#    context.plot_awgs(sequence, ylim=(-0.0,0.100), xlim=(0, 50))

    return context.run('1ns', sequence, m_param)


#%%
if __name__ == '__main__':
    ds1 = test1()
