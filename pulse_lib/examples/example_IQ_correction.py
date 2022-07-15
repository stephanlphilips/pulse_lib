import matplotlib.pyplot as pt
from numpy import pi
import qcodes as qc

from pulse_lib.tests.hw_schedule_mock import HardwareScheduleMock



from utils.plot import plot_awgs
from configuration.medium_iq import init_hardware, init_pulselib

#import qcodes.logger as logger
from qcodes.logger import start_all_logging

start_all_logging()


awgs, digitizer = init_hardware()
p = init_pulselib(awgs, digitizer)

# set qubit frequencies for nice figures

p.set_qubit_idle_frequency('q1', 2.420e9)
p.set_qubit_idle_frequency('q2', 2.440e9)
p.set_qubit_idle_frequency('q3', 2.840e9)
p.set_qubit_idle_frequency('q4', 2.840e9)

p.set_qubit_correction_gain('q2', 0.9, 1.0)
p.set_qubit_correction_gain('q3', 1.0, 0.9)
p.set_qubit_correction_phase('q4', 0.2*pi)

f_drive1 = 2.420e9
f_drive2 = 2.440e9
f_drive3 = 2.840e9
f_drive4 = 2.840e9
t_X90 = 100
amplitude = 100
t_wait = 50

manip = p.mk_segment()

manip.q1.add_MW_pulse(0, t_X90, amplitude, f_drive1)
manip.q1.wait(t_wait)
manip.reset_time()
manip.q1.add_MW_pulse(0, t_X90, amplitude, f_drive1)
manip.q2.add_MW_pulse(0, t_X90, amplitude, f_drive2)
manip.q1.wait(t_wait)
manip.reset_time()
manip.q1.add_MW_pulse(0, t_X90, amplitude, f_drive1)
manip.q3.add_MW_pulse(0, t_X90, amplitude, f_drive3)
manip.q1.wait(t_wait)
manip.reset_time()
manip.q1.add_MW_pulse(0, t_X90, amplitude, f_drive1)
manip.q4.add_MW_pulse(0, t_X90, amplitude, f_drive4)

# generate the sequence from segments
my_seq = p.mk_sequence([manip])
my_seq.set_hw_schedule(HardwareScheduleMock())
my_seq.n_rep = 1
my_seq.sample_rate = 1e9

my_seq.upload()
my_seq.play()

plot_awgs(awgs)
pt.legend()
pt.grid(True)

