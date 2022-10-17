import matplotlib.pyplot as pt

from pulse_lib.tests.hw_schedule_mock import HardwareScheduleMock

from configuration.medium_iq import init_hardware, init_pulselib
from utils.plot import plot_awgs



# create "AWG1"
awgs, digs = init_hardware()

# create channels P1, P2
p = init_pulselib(awgs, digs, virtual_gates=True)


seg1 = p.mk_segment()

seg1.P1.wait(500)
seg1.P1.add_block(100, 200, 100)
seg1.vP2.add_block(200, 300, 100)
seg1.reset_time()
seg1.q1.add_MW_pulse(80, 120, amp=40, freq=2.425e9)

seg2 = p.mk_segment()

seg2.P3.wait(500)
seg2.P3.add_block(150, 250, -50)
seg2.vP4.add_block(250, 350, -50)
seg2.reset_time()
seg2.q2.add_MW_pulse(50, 100, amp=80, freq=2.450e9)

seg12 = seg1 + seg2

pt.figure()
pt.title('segment 1')
seg1.plot()

pt.figure()
pt.title('segment 2')
seg2.plot()

pt.figure()
pt.title('segment 1+2')
seg12.plot()

# create sequence
seq = p.mk_sequence([seg12])
seq.set_hw_schedule(HardwareScheduleMock())
seq.n_rep = 1


seq.upload()
seq.play()


plot_awgs(awgs)
pt.title('AWG upload with DC compensation pulse at end')
