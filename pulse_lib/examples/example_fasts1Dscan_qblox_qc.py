import qcodes

from qcodes.data.io import DiskIO
from qcodes.measure import Measure

from configuration.small import init_hardware, init_pulselib

from pulse_lib.fast_scan.qblox_fast_scans import fast_scan1D_param

path = 'C:/Projects/test_data'
io = DiskIO(path)
qcodes.data.data_set.DataSet.default_io = io

awgs, digs = init_hardware()

# create channels P1, P2, vP1, vP2, SD1, SD2
p = init_pulselib(awgs, digs, virtual_gates=True)

# Reading param uploads, plays and returns data
fast_param = fast_scan1D_param(p, gate='vP1', swing=100.0, n_pt=50, t_step=5000)

m = Measure(fast_param)
ds = m.run()

