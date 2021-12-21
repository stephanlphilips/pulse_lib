import matplotlib.pyplot as pt
import numpy as np

from pulse_lib.tests.hw_schedule_mock import HardwareScheduleMock

from configuration.small import init_hardware, init_pulselib
from utils.plot import plot_awgs

from pulse_lib.virtual_channel_constructors import add_detuning_channels, virtual_gates_constructor

# create "AWG1"
awgs, digs = init_hardware()

# create channels P1, P2
p = init_pulselib(awgs, digs, virtual_gates=True)

custom_detuning = False

if custom_detuning:
    # set a virtual gate matrix
    detuning_gate_set = virtual_gates_constructor(p, 'detuning12', matrix_virtual2real=True)
    detuning_gate_set.add_real_gates('P1', 'P2')
    detuning_gate_set.add_virtual_gates('e12','U12')
    matrix = np.array([[+0.5, +1.0], [-0.5, +1.0]])
    detuning_gate_set.add_virtual_gate_matrix(matrix)
else:
    add_detuning_channels(p, 'P1', 'P2', 'e12', 'U12')

add_detuning_channels(p, 'vP1', 'vP2', 've12', 'vU12')


seg1 = p.mk_segment()

s = seg1
s.P1.add_block(52, 200, 400)
s.P2.add_block(252, 400, 400)
s.P2.wait(200)

seg2 = p.mk_segment()
s = seg2
s.e12.add_block(52, 200, 400)
s.U12.add_block(252, 400, 400)
s.U12.wait(200)
s.reset_time()
s.U12.add_block(52, 252, 400)
s.e12.add_block(152, 252, 200)
s.e12.wait(200)

seg3 = p.mk_segment()
s = seg3
s.vP1.add_block(52, 200, 400)
s.vP2.add_block(252, 400, 400)
s.vP2.wait(200)

seg4 = p.mk_segment()
s = seg4
s.ve12.add_block(52, 200, 400)
s.vU12.add_block(252, 400, 400)
s.vU12.wait(200)
s.reset_time()
s.vU12.add_block(52, 252, 400)
s.ve12.add_block(152, 252, 200)
s.ve12.wait(200)

# generate the sequence from segments
my_seq = p.mk_sequence([seg1, seg2, seg3, seg4])
my_seq.set_hw_schedule(HardwareScheduleMock())
my_seq.n_rep= 3

my_seq.upload()

my_seq.play()

pt.figure()
seg2.e12.plot_segment()
seg2.U12.plot_segment()
seg2.P1.plot_segment()
seg2.P2.plot_segment()
pt.title('detuning and real channels')
pt.grid(True)

pt.figure()
seg4.ve12.plot_segment()
seg4.vU12.plot_segment()
seg4.vP1.plot_segment()
seg4.vP2.plot_segment()
seg4.P1.plot_segment()
seg4.P2.plot_segment()
pt.title('detuning, virtual and real channels')
pt.grid(True)

plot_awgs(awgs)
pt.title('AWG upload (with DC compensation)')
pt.grid(True)
