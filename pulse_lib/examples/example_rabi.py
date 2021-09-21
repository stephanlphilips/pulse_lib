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

t_X90 = 50
amplitude = 50

t_pulse = lp.linspace(100, 1000, 10, axis=0)
f_drive = lp.linspace(2.41e9, 2.43e9, 11, axis=1)
amplitude = 50

# init pulse
init = p.mk_segment()
init.add_block(0, 100, gates, v_init, reset_time=True)
init.add_ramp(0, 50, gates, v_init, v_manip)

manip = p.mk_segment()
manip.q1.add_MW_pulse(0, t_pulse, amplitude, f_drive)

# read-out
t_measure = 200 # short time for visibility of pulse
readout = p.mk_segment()
readout.add_ramp(0, 100, gates, v_manip, v_read, reset_time=True)
readout.add_block(0, t_measure, gates, v_read)
readout.SD1.acquire(0, t_measure)

# generate the sequence from segments
my_seq = p.mk_sequence([init, manip, readout])
my_seq.set_hw_schedule(HardwareScheduleMock())
my_seq.n_rep = 1
my_seq.sample_rate = 1e9


for f in [0, 9]:
    for t in [0, 1, 9]:
        my_seq.upload([f,t])
        my_seq.play([f,t])
        plot_awgs(awgs)
        pt.grid(True)
        pt.title(f'f={f_drive[f]/1e6:7.2f} MHz, t={t_pulse[t]} ns')




