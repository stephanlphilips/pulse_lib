import matplotlib.pyplot as pt

from pulse_lib.tests.hw_schedule_mock import HardwareScheduleMock

from configuration.small import init_hardware, init_pulselib
from utils.plot import plot_awgs


# create "AWG1"
awgs = init_hardware()

# create channels P1, P2
p = init_pulselib(awgs, virtual_gates=True)


seg1 = p.mk_segment()

s = seg1
s.P1.add_ramp_ss(0, 400, 0, 1000)
s.P1.add_block(400, 600, 1000)
s.P1.add_ramp_ss(600, 1000, 1000, 500)
s.P2.add_block(1200, 1500, 500)
s.P2.wait(500)

seg2 = p.mk_segment()
s = seg2
s.vP1.add_ramp_ss(0, 400, 0, 1000)
s.vP1.add_block(400, 600, 1000)
s.vP1.add_ramp_ss(600, 1000, 1000, 500)
s.vP2.add_block(1200, 1500, 500)
s.vP2.wait(500)

# generate the sequence from segments
my_seq = p.mk_sequence([seg1, seg2])
my_seq.set_hw_schedule(HardwareScheduleMock())

my_seq.upload()

my_seq.play()

seg2.vP1.plot_segment()
seg2.vP2.plot_segment()
seg2.P1.plot_segment()
seg2.P2.plot_segment()
pt.title('virtual and real channels')
pt.grid(True)

plot_awgs(awgs)
pt.title('AWG upload (with DC compensation)')
pt.grid(True)
