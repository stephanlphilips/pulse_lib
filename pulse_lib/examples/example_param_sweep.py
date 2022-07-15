import matplotlib.pyplot as pt

from pulse_lib.tests.hw_schedule_mock import HardwareScheduleMock
import pulse_lib.segments.utility.looping as lp

from configuration.small import init_hardware, init_pulselib
from utils.plot import plot_awgs


# create "AWG1"
awgs, digs = init_hardware()

# create channels P1, P2
p = init_pulselib(awgs, digs)

v_param = lp.linspace(0, 200, 5, axis=0, unit = "mV", name = "vPulse")
t_wait = lp.linspace(20, 100, 3, axis=1, unit = "mV", name = "t_wait")


seg1 = p.mk_segment()
seg2 = p.mk_segment()

seg1.P1.add_ramp_ss(0, 100, 0, v_param)
seg1.P1.add_block(100, 200, v_param)

seg2.P2.add_block(0, 100, 200)
seg2.P2.wait(t_wait)
seg2.reset_time()
seg2.SD1.acquire(40)
seg2.SD1.wait(1000)
seg2.P1.add_block(0, 100, v_param)

# create sequence
seq = p.mk_sequence([seg1,seg2])
seq.n_rep = 10
seq.set_hw_schedule(HardwareScheduleMock())
seq.set_acquisition(t_measure=100)

for t in seq.t_wait.values:
    for v_pulse in seq.vPulse.values:
        seq.t_wait(t)
        seq.vPulse(v_pulse)
        seq.upload()
        seq.play()

        plot_awgs(awgs)
        pt.title('AWG upload {t} - {v_pulse}')
