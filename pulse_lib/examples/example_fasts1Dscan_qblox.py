import matplotlib.pyplot as pt

from configuration.small import init_hardware, init_pulselib

from utils.plot import plot_awgs
from pulse_lib.fast_scan.qblox_fast_scans import fast_scan1D_param

awgs, digs = init_hardware()

# create channels P1, P2, vP1, vP2, SD1, SD2
p = init_pulselib(awgs, digs, virtual_gates=True)

fast_param = fast_scan1D_param(p, gate='vP1', swing=100.0, n_pt=50, t_step=5000, n_avg=3)

# Reading param uploads, plays and returns data
data = fast_param()

pt.figure()
names = fast_param.names
for ch_name,values in zip(fast_param.names, data):
    print(ch_name, values)
    pt.plot(values, label=ch_name)
pt.legend()


plot_awgs(awgs+digs)
pt.title('AWG upload')
pt.grid(True)

#%% Example qcodes Measurement (Legacy)
import qcodes

from qcodes.data.io import DiskIO
from qcodes.measure import Measure

path = 'C:/Projects/test_data'
io = DiskIO(path)
qcodes.data.data_set.DataSet.default_io = io


m = Measure(fast_param)
ds = m.run()

