
from pulse_lib.tests.configurations.test_configuration import context

#%%
import pulse_lib.segments.utility.looping as lp
import numpy as np


def get_min_sample_rate(duration):
    # Keysight minimum 2000 samples in segment,
    # but due to segment 'welding' the segment should be a bit longer
    # at least 2000 ns at start and 10 samples at end.
    for sr in [1e6, 2e6, 5e6, 1e7, 2e7, 5e7, 1e8]:
        if (duration-2000) * 1e-9 * sr > 2000 + 10:
            return sr
    return 1e9


def test():
    pulse = context.init_pulselib(n_gates=2)

    t_wait = lp.geomspace(1000, 100000, 5, 't_wait', unit='ns', axis=0)

    s = pulse.mk_segment()

    calc_sr = np.frompyfunc(get_min_sample_rate, 1, 1)
    sr = calc_sr(t_wait)
    # print(sr)
    s.sample_rate = sr

    s.P1.add_block(0, 100, 80.0)
    s.wait(t_wait, reset_time=True)
    s.P1.add_block(0, 100, 80.0)

    sequence = pulse.mk_sequence([s])
    context.add_hw_schedule(sequence)
    for t in sequence.t_wait.values:
        sequence.t_wait(t)
        context.plot_awgs(sequence, ylim=(-0.100,0.100))

    return None

if __name__ == '__main__':
    ds = test()
