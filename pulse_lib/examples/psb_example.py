import matplotlib.pyplot as pt

from pulse_lib.tests.hw_schedule_mock import HardwareScheduleMock
import pulse_lib.segments.utility.looping as lp

from configuration.small import init_hardware, init_pulselib

from utils.plot import plot_awgs


# create "AWG1"
awgs, digs = init_hardware()

# create channels P1, P2
p = init_pulselib(awgs, digs, virtual_gates=True)

seg  = p.mk_segment()

#  points to pulse to
gates = ['vP1', 'vP2']
P0 = (-100, -100)
P1 = (-100,  100)
P2 = ( -25,   25)
P3 = (   5,   -5)

# check PSB result for all values in square defined by P2 and P3
PSB_sweep = (
        lp.linspace(P2[0],P3[0],5, axis=0, unit = "mV", name = "vP1"),
        lp.linspace(P2[1],P3[1],10, axis=1, unit = "mV", name = "vP2")
        )


seg.add_block(0, 10000, gates, P0, reset_time=True)
seg.add_block(0, 10000, gates, P1, reset_time=True)
seg.add_ramp(0, 500, gates, P1, P2, reset_time=True)
seg.add_ramp(0, 500, gates, P2, PSB_sweep, reset_time=True)
seg.add_HVI_marker('dig_trigger_1', 100)
seg.add_block(0, 50000, gates, PSB_sweep, reset_time=True)
seg.add_block(0, 10000, gates, P0, reset_time=True)


for index in [(0,0), (0,4), (9,0), (9,4)]:
    pt.figure()
    seg.plot(index=index, channels=['vP1', 'vP2', 'P1', 'P2'])

# generate the sequence from segments
my_seq = p.mk_sequence([seg])
my_seq.set_hw_schedule(HardwareScheduleMock())

my_seq.upload((9,4))

my_seq.play((9,4))

plot_awgs(awgs)
pt.title('AWG upload (with DC compensation)')
pt.grid(True)
