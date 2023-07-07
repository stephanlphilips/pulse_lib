
from pulse_lib.tests.configurations.test_configuration import context

#%%
import numpy as np

def test_sin(t_start=0, duration=10, frequency=100e6):
    pulse = context.init_pulselib(n_gates=1)

    segments = []

    s = pulse.mk_segment()
    segments.append(s)

    s.wait(0, reset_time=True)
    s.reset_time()

    s.P1.add_sin(t_start, t_start+duration, 100, frequency)

    s.reset_time()
    s.wait(10)

    sequence = pulse.mk_sequence(segments)
    sequence.n_rep = None

    context.plot_segments(segments)
    context.plot_awgs(sequence, ylim=(-0.2, 0.2))

def test_ramp_sin(t_ramp=11, duration=50, frequency=20e6):
    pulse = context.init_pulselib(n_gates=1)

    segments = []

    s = pulse.mk_segment()
    segments.append(s)

    s.wait(20, reset_time=True)
    s.reset_time()

    s.P1.add_ramp_ss(0, t_ramp, 0, 100)
    s.reset_time()
    s.P1.add_sin(0, duration, 100, frequency, np.pi/2)
    s.reset_time()
    s.P1.add_ramp_ss(0, t_ramp, 100, 0)

    s.reset_time()
    s.wait(10)

    sequence = pulse.mk_sequence(segments)
    sequence.n_rep = None

    context.plot_awgs(sequence, ylim=(-0.2, 0.2))

def test_sin_on_ramp(t_ramp=50, t_start=15, duration=20, frequency=100e6):
    pulse = context.init_pulselib(n_gates=1)

    segments = []

    s = pulse.mk_segment()
    segments.append(s)

    s.wait(20, reset_time=True)
    s.reset_time()

    s.P1.add_ramp_ss(0, t_ramp, 0, 100)
    s.P1.add_sin(t_start, t_start+duration, 100, frequency)

    s.reset_time()
    s.wait(10)

    sequence = pulse.mk_sequence(segments)
    sequence.n_rep = None

    context.plot_awgs(sequence, ylim=(-0.2, 0.2))

def test_multiple_sin(t_ramp=50, t_start=15, duration=65, frequency=100e6):
    pulse = context.init_pulselib(n_gates=1)

    segments = []

    s = pulse.mk_segment()
    segments.append(s)

    s.wait(20, reset_time=True)
    s.reset_time()

#    s.P1.add_ramp_ss(0, t_ramp, 0, 10)
    s.P1.add_sin(t_start, t_start+duration, 100, frequency)
    s.P1.add_sin(t_start, t_start+duration, 50, frequency/2, np.pi/4)

    s.reset_time()
    s.wait(10)

    sequence = pulse.mk_sequence(segments)
    sequence.n_rep = None

    context.plot_awgs(sequence, ylim=(-0.2, 0.2))

def test_long_repetitive_sines(t_ramp=50, t_start=15, duration=605, frequency=10e6):
    pulse = context.init_pulselib(n_gates=1)

    segments = []

    s = pulse.mk_segment()
    segments.append(s)

    s.wait(20+t_start, reset_time=True)
    s.reset_time()

    period = int(1e9/frequency)*2
    n_periods = duration // period
    rem = duration % period
    for _ in range(n_periods):
        s.P1.add_sin(0, period, 100, frequency)
        s.P1.add_sin(0, period, 50, frequency/2, np.pi/4)
        s.reset_time()

    s.P1.add_sin(0, rem, 100, frequency)
    s.P1.add_sin(0, rem, 50, frequency/2, np.pi/4)
    s.reset_time()
    s.wait(10)

    sequence = pulse.mk_sequence(segments)
    sequence.n_rep = None

    context.plot_awgs(sequence, ylim=(-0.2, 0.2))


#%%
if __name__ == '__main__':
    test_sin()
    test_sin(t_start=1)
    test_sin(t_start=2)
    test_sin(t_start=2, duration=200)
    test_ramp_sin()
    test_sin_on_ramp()
    test_multiple_sin()
    test_long_repetitive_sines(t_start=0)
    test_long_repetitive_sines(t_start=2, duration=1850)
    test_long_repetitive_sines(t_start=2, duration=16_501, frequency=1e6)

#%%
if False:
    from pulse_lib.tests.utils.last_upload import get_last_upload

    lu = get_last_upload(context.pulse)