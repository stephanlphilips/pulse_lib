from pulse_lib.base_pulse import pulselib

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import pulse_lib.segments.utility.looping as lp

def init_pulse_lib():
    # minimalistic init
	pulse = pulselib()
	pulse.add_awgs('AWG1',None)
	pulse.define_channel('B2','AWG1', 1)
	pulse.define_channel('B4','AWG1', 2)

	pulse.finish_init()

	return pulse

pulse = init_pulse_lib()


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


seg  = pulse.mk_segment()

seg.B2.wait(10)
seg.B2.reset_time()
seg.B2.add_custom_pulse(0, 60, tukey_pulse, amplitude=35.0, alpha=0.5)
seg.B2.add_sin(20, 40, 5.0, 2e8)
seg.B2.wait(10)

# looping on arguments
alpha_loop = lp.linspace(0.3, 0.5, n_steps = 2, name = 'alpha', axis = 0)
amplitude_loop = lp.linspace(30, 50, n_steps = 3, name = 'amplitude', unit = 'mV', axis = 1)

seg.B4.wait(10)
seg.B4.reset_time()
seg.B4.add_custom_pulse(0, 60, tukey_pulse, amplitude=amplitude_loop, alpha=alpha_loop)
seg.B4.add_sin(20, 40, 5.0, 2e8+amplitude_loop)
seg.B4.wait(10)


plt.figure(2)
seg.B2.plot_segment([0])
plt.xlabel("time (ns)")
plt.ylabel("voltage (mV)")
plt.show()

plt.figure(4)
for i in range(2):
    for j in range(3):
        seg.B4.plot_segment([j,i])
plt.xlabel("time (ns)")
plt.ylabel("voltage (mV)")
plt.show()
