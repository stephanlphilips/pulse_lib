
import numpy as np
import matplotlib.pyplot as pt
from scipy.signal import gaussian

import pulse_lib.segments.utility.looping as lp
from pulse_lib.tests.hw_schedule_mock import HardwareScheduleMock

from configuration.small import init_hardware, init_pulselib, _backend
from utils.plot import plot_awgs


def gaussian_step(duration, sample_rate, amplitude, stddev):
    """
    Generates Gaussian step. The actual step is in the middle of the pulse.
    The slope in the middle is ~0.40/stddev*amplitude [mV/ns]

    Args:
        duration: time in ns of the pulse.
        sample_rate: sampling rate of the pulse (Sa/s).
        amplitude: amplitude of the pulse
        stddev: stddev of the Gaussian

    Example:
        # pulse with Gaussian ramp up and down.
        amp = 20.0
        start = 100 # ns
        stop = 200 # ns
        m = 20 # ns
        stddev = 4 # ns
        # step up
        gate.add_custom_pulse(start-m/2, start+m/2, amp, gaussian_step, stddev)
        # block starts after gaussian ramp up, but ends *after* ramp down.
        gate.add_block(start+m/2, stop+m/2, amp)
        gate.add_custom_pulse(stop-m/2, stop+m/2, -amp, gaussian_step, stddev)

    Returns:
        pulse (np.ndarray) : Tukey pulse
    """
    n_points = int(round(duration / sample_rate * 1e9))
    g = gaussian(n_points, stddev)
    gg = np.cumsum(g)
    gg /= gg[-1]
    m = n_points//2
    dy = gg[m]-gg[m-1]
    print(dy, dy*stddev, 0.40/stddev*amplitude)
    return gg * amplitude



if _backend == 'Qblox':
    raise Exception('This example does not yet work with Qblox. '
                    'Custom pulse and ramp may not overlap.')



# create "AWG1"
awgs, digs = init_hardware()

# create channels P1, P2
p = init_pulselib(awgs, digs, virtual_gates=True)

seg  = p.mk_segment()

gate = seg['P1']
amp = 500.0
start = 100 # ns
stop = 200 # ns
m = 80 # ns Should be at least 2*stddev
stddev = 10 # ns

# step up
gate.add_custom_pulse(start-m, start+m, amp, gaussian_step, stddev=stddev)
# block starts after gaussian ramp up, but ends *after* ramp down.
gate.add_block(start+m, stop+m, amp)
gate.add_custom_pulse(stop-m, stop+m, -amp, gaussian_step, stddev=stddev)
gate.wait(50-m)

# reset time aligns all channels after last pulse or wait.
seg.reset_time()

# looping on arguments
stddev_loop = lp.linspace(10, 2.5, n_steps=4, name='alpha', axis=0)
amplitude_loop = lp.linspace(100, 150, n_steps=2, name='amplitude', unit='mV', axis=1)

gate = seg.P2
gate.add_ramp_ss(0, stop, 0, 200)
# only step up
gate.add_custom_pulse(start-m, start+m, amplitude_loop, gaussian_step, stddev=stddev_loop)
# block starts after gaussian ramp up
gate.add_block(start+m, stop, amplitude_loop)
# add plateau after ramp till end of segment
gate.add_block(stop, -1, amplitude_loop+200)
gate.wait(50-m)

# fast straight ramp on P1 with same max slope as Gaussian on P2
gate = seg.P1
gate.add_ramp_ss(0, stop, 0, 200)
ramp_duration = stddev_loop*2.5
r = ramp_duration/2
gate.add_ramp(start-r-1, start+r-1, amplitude_loop)
# block starts after ramp
gate.add_block(start+r-1, stop, amplitude_loop)
# add plateau after ramp till end of segment
gate.add_block(stop, -1, amplitude_loop+200)


# create sequence
seq = p.mk_sequence([seg])
seq.set_hw_schedule(HardwareScheduleMock())

seq.upload((0,0))
seq.play((0,0))

plot_awgs(awgs)
pt.title('AWG upload with DC compensation pulse at end')


pt.figure()
seg.P1.plot_segment([0,0])

pt.figure(4)
pt.xlim(320, 380)
pt.ylim(0, 300)
pt.title('variation of amplitude')
for j in range(len(amplitude_loop)):
    seg.P2.plot_segment([j,0])

pt.figure(5)
pt.xlim(320, 380)
pt.ylim(0, 300)
pt.title('variation of stddev')
for i in range(len(stddev_loop)):
    seg.P2.plot_segment([0,i])
