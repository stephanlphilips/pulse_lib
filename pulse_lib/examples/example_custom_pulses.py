
import numpy as np
from scipy import signal
import matplotlib.pyplot as pt
import pulse_lib.segments.utility.looping as lp
from pulse_lib.tests.hw_schedule_mock import HardwareScheduleMock

from configuration.small import init_hardware, init_pulselib
from utils.plot import plot_awgs


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

def iswap_pulse(duration, sample_rate, amplitude, frequency, mw_amp):
    """
    Generates iswap pulse

    Args:
        duration: time in ns of the pulse.
        sample_rate: sampling rate of the pulse (Sa/s).
        amplitude: amplitude of the pulse
        frequency: frequency of sine
        mw_amp: amplitude of sine

    Returns:
        pulse (np.ndarray) : Tukey pulse
    """
    alpha=0.5
    n_points = int(round(duration / sample_rate * 1e9))
    n_mod = int(n_points * (1-alpha))
    n_pre = (n_points-n_mod) // 2

    pulse = signal.windows.tukey(n_points, alpha) * amplitude
    t_sin = np.arange(n_mod)/sample_rate
    pulse[n_pre:n_pre+n_mod] += mw_amp * signal.windows.tukey(n_mod, alpha) * np.sin(2*np.pi*frequency*t_sin)
    return pulse


# create "AWG1"
awgs, digs = init_hardware()

# create channels P1, P2
p = init_pulselib(awgs, digs, virtual_gates=True)

seg  = p.mk_segment()

seg.P1.wait(20)
seg.P1.reset_time()
seg.P1.add_custom_pulse(0, 60, 350.0, tukey_pulse, alpha=0.5)
#seg.P1.add_sin(20, 40, 50.0, 2e8)
seg.P1.wait(20)
seg.reset_time()

# looping on arguments
alpha_loop = lp.linspace(0.3, 0.5, n_steps = 2, name = 'alpha', axis = 0)
amplitude_loop = lp.linspace(300, 500, n_steps = 3, name = 'amplitude', unit = 'mV', axis = 1)

seg.P2.wait(20)
seg.P2.reset_time()
seg.P2.add_custom_pulse(0, 60, amplitude_loop, tukey_pulse, alpha=alpha_loop)
#seg.P2.add_sin(20, 40, 50.0, 2e8)
seg.P2.wait(20)
seg.reset_time()

# virtual gate: compensation is visible on P1
seg.vP2.add_custom_pulse(20, 80, 150.0, iswap_pulse, frequency=2e8, mw_amp=2.5)
seg.vP2.wait(20)

# create sequence
seq = p.mk_sequence([seg])
seq.set_hw_schedule(HardwareScheduleMock())

seq.upload()
seq.play()

plot_awgs(awgs)
pt.title('AWG upload with DC compensation pulse at end')


pt.figure()
seg.P1.plot_segment()

pt.figure(4)
for i in range(2):
    for j in range(3):
        seg.P2.plot_segment([j,i])
