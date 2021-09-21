from scipy.signal import windows
import matplotlib.pyplot as pt

from pulse_lib.tests.hw_schedule_mock import HardwareScheduleMock

from configuration.small import init_hardware, init_pulselib
from utils.plot import plot_awgs

# custom pulse
def tukey_pulse(duration, sample_rate, amplitude, alpha):
    n_points = int(round(duration / sample_rate * 1e9))
    return windows.tukey(n_points, alpha) * amplitude



# create "AWG1"
awgs = init_hardware()

# create channels P1, P2
p = init_pulselib(awgs)

seg1 = p.mk_segment()
seg2 = p.mk_segment()

seg1.P1.add_block(100, 200, 340)
seg1.P1.add_ramp_ss(200, 300, 340, 120)
# reset relative time base for channel P1
seg1.P1.reset_time()
# overlapping pulses are summed
seg1.P1.add_block(0, 200, 120)
seg1.P1.add_ramp_ss(180, 190, 0, 50)
seg1.P1.add_ramp_ss(190, 200, 50, 0)
seg1.P1.add_sin(50, 150, 50, 100e6)
# wait 100 ns after last pulse of segment
seg1.P1.wait(100)

# add voltage offset to segment
seg2.P1 += -200
#
seg2.P2.add_block(100, 180, 200)
seg2.P2.add_ramp_ss(180, 200, 200, 100)
seg2.P2.add_ramp_ss(200, 250, 100, 0)
seg2.P1.add_ramp_ss(100, 200, 0, -200)
# reset time base for SEGMENT: aligns all segments
seg2.reset_time()

seg2.P1.add_custom_pulse(0, 80, 142.0, tukey_pulse, alpha=0.7)

# alternative channel addressing:
seg2['P2'].add_block(60, 80, 50)
seg2['P2'].wait(100)

# create sequence
seq = p.mk_sequence([seg1,seg2])
seq.set_hw_schedule(HardwareScheduleMock())

seq.upload()
seq.play()

pt.figure()
pt.title('segment 1')
seg1.plot()

pt.figure()
pt.title('segment 2')
seg2.plot()

plot_awgs(awgs)
pt.title('AWG upload with DC compensation pulse at end')
