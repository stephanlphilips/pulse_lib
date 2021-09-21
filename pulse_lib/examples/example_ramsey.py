import numpy as np
import matplotlib.pyplot as pt

from pulse_lib.tests.hw_schedule_mock import HardwareScheduleMock
import pulse_lib.segments.utility.looping as lp

from configuration.small_iq import init_hardware, init_pulselib

from utils.plot import plot_awgs

# create "AWG1"
awgs = init_hardware()


# create channels P1, P2
p = init_pulselib(awgs, virtual_gates=True)

gates = ['vP1','vP2']
v_init = [70, 20]
v_manip = [0,0]
v_read = [30, 25]
t_measure = 100 # short time for visibility of other pulses

f_drive = 2.420e9
t_X90 = 50
amplitude = 50

t_wait = lp.linspace(0, 200, 21, axis=0)

# init pulse
init = p.mk_segment()
init.add_block(0, 100, gates, v_init)
init.add_ramp(100, 140, gates, v_init, v_manip)

# Ramsey
manip = p.mk_segment()
init.add_block(0, -1, gates, v_manip)
manip.q1.add_MW_pulse(0, t_X90, amplitude, f_drive)
manip.q1.wait(t_wait)
manip.q1.reset_time()
manip.q1.add_MW_pulse(0, t_X90, amplitude, f_drive)

# read-out
readout = p.mk_segment()
readout.add_ramp(0, 100, gates, v_manip, v_read)
readout.reset_time()
readout.add_block(0, t_measure, gates, v_read)
readout.SD1.acquire(0, t_measure)

# generate the sequence from segments
my_seq = p.mk_sequence([init, manip, readout])
my_seq.set_hw_schedule(HardwareScheduleMock())
my_seq.n_rep = 1000
my_seq.sample_rate = 1e9

for t in [0, 1, 2, 3, 10]:
    my_seq.upload([t])
    my_seq.play([t])

    plot_awgs(awgs)
    pt.title(f't={t_wait[t]} ns')
    pt.legend()
    pt.grid(True)

##     plot reference sine
#    t = np.arange(2*t_X90+t_wait[t]+100+t_measure)
#    s = np.cos(2*np.pi*20e6*(t)*1e-9)*0.050
#    pt.plot(t, s, ':', color='gray', label='ref')

#from pprint import pprint
#pprint(my_seq.metadata)