
from pulse_lib.tests.configurations.test_configuration import context

#%%
from scipy import signal


def tukey_pulse(duration, sample_rate, amplitude, alpha):
    """
    Generates Tukey shaped pulse

    Args:
        duration: time in ns of the pulse.
        sample_rate: sampling rate of the pulse (Sa/s).
        amplitude: amplitude of the pulse
        alpha: alpha coefficient of the Tukey window

    Returns:
        pulse (np.ndarray) : Tukey pulse
    """
    n_points = int(round(duration / sample_rate * 1e9))
    return signal.windows.tukey(n_points, alpha) * amplitude

#%%

def test1():
    pulse = context.init_pulselib(n_gates=1, n_markers=1)

    segments = []

    s = pulse.mk_segment()
    segments.append(s)

    s.M1.add_marker(0, 8)
    s.wait(10, reset_time=True)

    s.P1.add_custom_pulse(15, 75, 100.0, tukey_pulse, alpha=0.5)

    s.wait(10, reset_time=True)

    sequence = pulse.mk_sequence(segments)
    sequence.n_rep = 2

    context.plot_awgs(sequence, ylim=(-0.2, 0.2))

def test2():
    pulse = context.init_pulselib(n_gates=1, n_markers=1)

    segments = []

    s = pulse.mk_segment()
    segments.append(s)

    s.M1.add_marker(0, 8)
    s.wait(10, reset_time=True)

    s.P1.add_block(0, 200, 20)
    s.P1.add_custom_pulse(15, 75, 100.0, tukey_pulse, alpha=0.5)

    s.wait(10, reset_time=True)

    sequence = pulse.mk_sequence(segments)
    sequence.n_rep = 2

    context.plot_awgs(sequence, ylim=(-0.2, 0.2))


def test3():
    pulse = context.init_pulselib(n_gates=1, n_markers=1)

    segments = []

    s = pulse.mk_segment()
    segments.append(s)

    s.M1.add_marker(0, 8)
    s.wait(10, reset_time=True)

    s.P1.add_ramp_ss(0, 200, -50, 50)
    s.P1.add_custom_pulse(15, 75, 100.0, tukey_pulse, alpha=0.5)

    s.wait(10, reset_time=True)

    sequence = pulse.mk_sequence(segments)
    sequence.n_rep = 2

    context.plot_awgs(sequence, ylim=(-0.2, 0.2))


def test4():
    pulse = context.init_pulselib(n_gates=1, n_markers=1)

    segments = []

    s = pulse.mk_segment()
    segments.append(s)

    s.M1.add_marker(0, 8)
    s.wait(10, reset_time=True)

    s.P1.add_ramp_ss(0, 50, -50, 50)
    s.P1.add_ramp_ss(50, 100, 50, -50)
    s.P1.add_custom_pulse(15, 75, 100.0, tukey_pulse, alpha=0.5)

    s.wait(10, reset_time=True)

    sequence = pulse.mk_sequence(segments)
    sequence.n_rep = 2

    context.plot_awgs(sequence, ylim=(-0.2, 0.2))

def test5():
    pulse = context.init_pulselib(n_gates=1, n_markers=1)

    segments = []

    s = pulse.mk_segment()
    segments.append(s)

    s.M1.add_marker(0, 8)
    s.wait(10, reset_time=True)

    s.P1.add_ramp_ss(0, 150, -50, 50)
    s.P1.add_custom_pulse(15, 75, 100.0, tukey_pulse, alpha=0.5)
    s.P1.add_sin(30, 60, 20.0, 100e6)

    s.wait(10, reset_time=True)
    s.P1.add_sin(30, 60, 20.0, 100e6)

    sequence = pulse.mk_sequence(segments)
    sequence.n_rep = 2

    context.plot_awgs(sequence, ylim=(-0.2, 0.2))

#%%
if __name__ == '__main__':
    ds1 = test1()
    ds2 = test2()
    ds3 = test3()
    ds4 = test4()
    ds5 = test5()

#%%
if False:
    from pulse_lib.tests.utils.last_upload import get_last_upload

    lu = get_last_upload(context.pulse)