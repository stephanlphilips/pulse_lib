
from pulse_lib.tests.configurations.test_configuration import context
import pulse_lib.segments.utility.looping as lp
import time

def test():
    pulse = context.init_pulselib(n_gates=1, n_sensors=1)

    t_pulse = lp.linspace(200, 600, 3, name='t_pulse', axis=0)
    amplitude = lp.linspace(100, 400, 4, name='amplitude', axis=1)

    s = pulse.mk_segment()

    t_measure = 100
    s.wait(20000)
    s.P1.add_block(80, t_pulse, amplitude)
    s.SD1.acquire(100, t_measure)
    s.SD1.acquire(300, t_measure)
    s.SD1.acquire(500, t_measure)

    context.plot_segments([s], index=(0,0))

    sequence = pulse.mk_sequence([s])
    sequence.n_rep = 10000
    context.add_hw_schedule(sequence)
    m_param = sequence.get_measurement_param()

    for t in sequence.t_pulse.values:
        for amp in sequence.amplitude.values:
            sequence.t_pulse(t)
            sequence.amplitude(amp)
            context.plot_awgs(sequence, ylim=(-0.100,0.100))
            # data must be retrieved from digitizer.
            data = m_param()

    ds = context.run('test_2D_acquire', sequence, m_param)

    return ds

if __name__ == '__main__':
    import numpy as np
    ds = test()
    print(np.mean(ds.SD1_1, axis=2))
    print(np.mean(ds.SD1_2, axis=2))
    print(np.mean(ds.SD1_3, axis=2))
