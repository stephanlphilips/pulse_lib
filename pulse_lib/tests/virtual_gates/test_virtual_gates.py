
from pulse_lib.tests.configurations.test_configuration import context
import matplotlib.pyplot as pt


#%%
def test1():
    pulse = context.init_pulselib(n_gates=2, n_sensors=1, virtual_gates=True)

    pulse.add_channel_attenuation('P1', 0.1)
    pulse.add_channel_attenuation('P2', 0.1)

    s = pulse.mk_segment()

    s.vP1.add_block(0,100, -10)
    s.vP1.add_ramp_ss(4, 8, 0, 80)
    s.vP1.add_block(8, 10, 80)
    s.vP1.add_ramp_ss(10, 14, 80, 0)
    s.vP2.wait(20)
    s.vP2.reset_time()
    s.vP2.add_ramp_ss(4, 8, 0, 80)
    s.vP2.add_block(8, 10, 80)
    s.vP2.add_ramp_ss(10, 14, 80, 0)

    s.vP1.add_ramp_ss(60, 90, 20, 40)
    s.vP1.add_ramp_ss(70, 80, 0, -40)
    s.reset_time()
    s.SD1.acquire(0, 100, wait=True)
#    s.wait(100000)

    sequence = pulse.mk_sequence([s])
#    sequence.n_rep = 10000
    sequence.n_rep = None
    m_param = sequence.get_measurement_param()

    context.plot_awgs(sequence, ylim=(-0.2,1.100), xlim=(0, 100))
    context.run('virtual_gates', sequence, m_param, close_sequence=False)

    context.virtual_matrix[1,0] = 0.2
    sequence.recompile()

    context.plot_awgs(sequence, ylim=(-0.2,1.100), xlim=(0, 100))
    return context.run('virtual_gates2', sequence, m_param)


#%%
if __name__ == '__main__':
    ds1 = test1()
