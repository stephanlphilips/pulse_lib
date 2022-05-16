import matplotlib.pyplot as pt

from configuration.small import init_hardware, init_pulselib

from utils.plot import plot_awgs
from pulse_lib.fast_scan.qblox_fast_scans import fast_scan2D_param

# create "AWG1"
awgs, digs = init_hardware()

# create channels P1, P2
p = init_pulselib(awgs, digs, virtual_gates=True)

fast_param = fast_scan2D_param(p,
                               gate1='P1', swing1=100.0, n_pt1=10,
                               gate2='P2', swing2=35.0, n_pt2=20,
                               t_step=4000)

# Reading the parameter uploads and plays the sequence and returns data
data = fast_param()

pt.figure()
names = fast_param.names
for ch_name,values in zip(fast_param.names, data):
    print(ch_name, values)
    pt.plot(values.flat, label=ch_name)
pt.legend()


plot_awgs(awgs+digs)
pt.title('AWG upload')
pt.grid(True)


