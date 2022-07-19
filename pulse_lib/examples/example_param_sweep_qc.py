from pulse_lib.tests.hw_schedule_mock import HardwareScheduleMock
import pulse_lib.segments.utility.looping as lp

from qcodes import Station
from qcodes.data.data_set import DataSet
from qcodes.data.io import DiskIO
from qcodes.measure import Measure
from qcodes.loops import Loop
from qcodes.actions import Task

from configuration.small import init_hardware, init_pulselib
#from utils.plot import plot_awgs

def upload_play(seq):
    seq.upload()
    seq.play()
    # plot_awgs(awgs)

# TODO: add as qc_utility to pulselib.
def qc_run(name, seq, *params):
    loop = None
    for sp in seq.params:
        if loop is None:
            loop = Loop(sp[sp.values])
        else:
            loop = loop.loop((sp[sp.values]))

    play_task = Task(upload_play, seq)

    if loop is not None:
        m = loop.each(play_task, *params)
    else:
        m = Measure(play_task, *params)

    ds = m.run(loc_record={'name':name})
    return ds


path = 'C:/Projects/test/data'

io = DiskIO(path)
DataSet.default_io = io

station = Station()

# create "AWG1"
awgs,digs = init_hardware()

# create channels P1, P2
p = init_pulselib(awgs, digs)

v_param = lp.linspace(0, 200, 5, axis=0, unit = "mV", name = "vPulse")
t_wait = lp.linspace(20, 100, 3, axis=1, unit = "mV", name = "t_wait")


seg1 = p.mk_segment()
seg2 = p.mk_segment()

seg1.P1.add_ramp_ss(0, 100, 0, v_param)
seg1.P1.add_block(100, 200, v_param)

seg2.P2.add_block(0, 100, 200)
seg2.P2.wait(t_wait)
seg2.reset_time()
seg2.SD1.acquire(40)
seg2.P1.add_block(0, 100, v_param)
seg2.SD1.wait(1000)

# create sequence
seq = p.mk_sequence([seg1,seg2])
seq.n_rep=5
seq.set_hw_schedule(HardwareScheduleMock())
seq.set_acquisition(t_measure=100)
param = seq.get_measurement_param()

qc_run('SweepDemo', seq, param)
