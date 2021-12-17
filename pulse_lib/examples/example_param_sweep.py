import matplotlib.pyplot as pt

from pulse_lib.tests.hw_schedule_mock import HardwareScheduleMock
import pulse_lib.segments.utility.looping as lp

from configuration.small import init_hardware, init_pulselib
from utils.plot import plot_awgs


# create "AWG1"
awgs, digs = init_hardware()

# create channels P1, P2
p = init_pulselib(awgs, digs)

v_param = lp.linspace(0, 200, 11, axis=0, unit = "mV", name = "vPulse")
t_wait = lp.linspace(20, 100, 5, axis=1, unit = "mV", name = "t_wait")


seg1 = p.mk_segment()
seg2 = p.mk_segment()

seg1.P1.add_ramp_ss(0, 100, 0, v_param)
seg1.P1.add_block(100, 200, v_param)

seg2.P2.add_block(0, 100, 200)
seg2.P2.wait(t_wait)
seg2.reset_time()
seg2.add_HVI_marker('dig_trigger_1', t_off=50)
seg2.P1.add_block(0, 100, v_param)

# create sequence
seq = p.mk_sequence([seg1,seg2])
seq.set_hw_schedule(HardwareScheduleMock())

for index in ([(0,8), (2,2)]):
    seq.upload(index=index)
    seq.play(index=index)

pt.figure()
pt.title('segment 1')
seg1.plot((0,10))

for index in [(0,0), (0,2), (4,8)]:
    pt.figure()
    pt.title(f'segment 2 - {index}')
    seg2.plot(index=index)


plot_awgs(awgs)
pt.title('AWG upload with DC compensation pulse at end')
