import matplotlib.pyplot as pt

from pulse_lib.tests.hw_schedule_mock import HardwareScheduleMock
import pulse_lib.segments.utility.looping as lp

from configuration.small_iq import init_hardware, init_pulselib

from utils.plot import plot_awgs

# create "AWG1"
awgs, digs = init_hardware()

# create channels P1, P2
p = init_pulselib(awgs, digs, virtual_gates=True)

gates = ['vP1','vP2']
v_init = [70, 20]
v_manip = [0,0]
v_read = [30, 25]
t_measure = 100 # short time for visibility of other pulses

t_X90 = 60
amplitude = 50

t_pulse = lp.linspace(200, 1000, 5, name='t', unit='ns', axis=0)
f_drive = lp.linspace(2.41e9, 2.43e9, 3, name='freq', unit='Hz', axis=1)
amplitude = 50

# init pulse
init = p.mk_segment()
init.add_block(0, 100, gates, v_init, reset_time=True)
init.add_ramp(0, 60, gates, v_init, v_manip)

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
my_seq.n_rep = 3
my_seq.sample_rate = 1e9

# optionally: set t_measure globablly and omit in acquire call()
# my_seq.set_acquisition(t_measure=t_measure)

for t in my_seq.t.values:
    my_seq.t(t)
    for freq in my_seq.freq.values:
        my_seq.freq(freq)
        my_seq.upload()
        my_seq.play()
        plot_awgs(awgs)
        pt.grid(True)
        pt.title(f'f={freq/1e6} MHz, t={t} ns')




