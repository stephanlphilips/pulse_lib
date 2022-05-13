import matplotlib.pyplot as pt

from pulse_lib.tests.hw_schedule_mock import HardwareScheduleMock

from configuration.small import init_hardware, init_pulselib
from utils.plot import plot_awgs


# create "AWG1"
awgs = init_hardware()

# RC time = 1 ms
bias_T_rc_time = 0.1
compensate_bias_T = True

# create channels P1, P2
p = init_pulselib(awgs, virtual_gates=True,
                  bias_T_rc_time=bias_T_rc_time if compensate_bias_T else None)


#  points to pulse to
gates = ['vP1', 'vP2']

P0 = (-200, -200)
P1 = (-200,  200)
P2 = ( -25,   25)
P3 = (  25,  -25)

seg1 = p.mk_segment(name='init')
seg1.add_block(0, 10000000, gates, P0, reset_time=True)
seg2 = p.mk_segment(name='load')
seg2.add_block(0, 10000000, gates, P1, reset_time=True)
seg3 = p.mk_segment(name='measure')
seg3.add_block(0, 50000000, gates, P3, reset_time=True)



# generate the sequence and upload it.
my_seq = p.mk_sequence([seg1, seg2, seg3])

my_seq.set_hw_schedule(HardwareScheduleMock())
my_seq.sample_rate = 1e7

my_seq.upload()

my_seq.play()

plot_awgs(awgs, bias_T_rc_time=bias_T_rc_time)
pt.title('With bias-T compensation')
pt.grid(True)
pt.ylim(-0.30, 0.55)

