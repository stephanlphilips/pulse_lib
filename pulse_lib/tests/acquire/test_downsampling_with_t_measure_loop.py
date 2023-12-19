
from pulse_lib.tests.configurations.test_configuration import context
import pulse_lib.segments.utility.looping as lp

#%%


def test():
    pulse = context.init_pulselib(n_gates=6, n_qubits=4, n_sensors=2)

    t_measure = lp.linspace(1000, 5_000, 5, name='t_measure', axis=0)

    s = pulse.mk_segment()

    s.SD1.acquire(100, -1)
    s.P1.add_block(100, 1000, 500)
    s.wait(t_measure)

    context.plot_segments([s], index=(0,0))

    sequence = pulse.mk_sequence([s])
    sequence.n_rep = 10
    sequence.set_acquisition(sample_rate=1000e3)
    m_param = sequence.get_measurement_param()
    print(m_param.setpoints)

    for t in sequence.t_measure.values:
        sequence.t_measure(t)
        context.plot_awgs(sequence, ylim=(-0.100,0.100))
        # data must be retrieved from digitizer.
        data = m_param()
        print(data)

    ds = context.run('downsampling_t_measure_loop', sequence, m_param)

    return ds


#%%
if __name__ == '__main__':
    import numpy as np
    ds = test()
#    print(np.mean(ds.SD1_1, axis=2))
#    print(np.mean(ds.SD1_2, axis=2))
#    print(np.mean(ds.SD1_3, axis=2))
