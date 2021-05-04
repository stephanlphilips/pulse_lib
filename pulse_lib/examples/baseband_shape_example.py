from pulse_lib.base_pulse import pulselib

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import pulse_lib.segments.utility.looping as lp
from pulse_lib.virtual_channel_constructors import virtual_gates_constructor

def init_pulse_lib():
    # minimalistic init
	pulse = pulselib()
	pulse.define_channel('B2','AWG1', 1)
	pulse.define_channel('B4','AWG1', 2)

	virtual_gate_set_1 = virtual_gates_constructor(pulse)
	virtual_gate_set_1.add_real_gates('B2','B4')
	virtual_gate_set_1.add_virtual_gates('vB2','vB4')
	virtual_gate_set_1.add_virtual_gate_matrix(np.eye(2)*0.9+0.1)

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


seg  = pulse.mk_segment()

seg.B2.wait(10)
seg.B2.reset_time()
seg.B2.add_custom_pulse(0, 60, 35.0, tukey_pulse, alpha=0.5)
seg.B2.add_sin(20, 40, 5.0, 2e8)
seg.B2.wait(30)

# looping on arguments
alpha_loop = lp.linspace(0.3, 0.5, n_steps = 2, name = 'alpha', axis = 0)
amplitude_loop = lp.linspace(30, 50, n_steps = 3, name = 'amplitude', unit = 'mV', axis = 1)

seg.B4.wait(10)
seg.B4.reset_time()
seg.B4.add_custom_pulse(0, 60, amplitude_loop, tukey_pulse, alpha=alpha_loop)
seg.B4.add_sin(20, 40, 5.0, 2e8)
seg.B4.wait(30)

# virtual gate: compensation is visible on B4
seg.vB2.add_custom_pulse(80, 140, 15.0, iswap_pulse, frequency=2e8, mw_amp=2.5)

seg.enter_rendering_mode()

plt.figure(2)
seg.B2.plot_segment([0])
#plt.xlabel("time (ns)")
#plt.ylabel("voltage (mV)")
plt.show()

plt.figure(4)
for i in range(2):
    for j in range(3):
        seg.B4.plot_segment([j,i])
#plt.xlabel("time (ns)")
#plt.ylabel("voltage (mV)")
plt.show()

seg.exit_rendering_mode()
