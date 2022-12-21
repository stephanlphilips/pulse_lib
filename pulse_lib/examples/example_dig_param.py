import logging
from pprint import pprint
import numpy as np
import matplotlib.pyplot as pt
from collections.abc import Sequence

import qcodes.logger as logger
from qcodes.logger import start_all_logging

from pulse_lib.tests.hw_schedule_mock import HardwareScheduleMock
#from projects.keysight_measurement.hvi2.hvi2_schedule_loader import Hvi2ScheduleLoader

from configuration.medium_iq import init_hardware, init_pulselib
from utils.plot import plot_awgs

#start_all_logging()
#logger.get_file_handler().setLevel(logging.DEBUG)


def create_seq(pulse_lib):

    seg1 = pulse_lib.mk_segment(name='init')
    s = seg1
    s.vP1.add_block(0, 2000, 50)
    s.vP2.add_ramp_ss(0, 100, 50, 100)
    s.vP2.add_ramp_ss(100, 200, 100, 50)

    seg2 = pulse_lib.mk_segment('manip')
    s = seg2
    s.vP2.add_ramp_ss(0, 100, 50, 100)
    s.vP2.add_ramp_ss(100, 2000, 100, 50)
    s.reset_time()
    s.q1.add_MW_pulse(100, 200, 50, 2.45e9)

    seg3 = pulse_lib.mk_segment('measure')
    s = seg3
    s.vP1.add_block(0, 1e4, -90)
    s.vP2.add_block(0, 1e4, 120)
    s.vP3.add_ramp_ss(1e4, 2e4, 0, 50)
    s.vP3.add_block(2e4, 3e4, 50)
    s.vP3.add_ramp_ss(3e4, 3.5e4, 50, 0)
    s.vP1.add_block(2e4, 3e4, -100)
    s.vP2.add_block(2e4, 3e4, 120)
    s.SD1.acquire(2e4, ref='m1', threshold=10, zero_on_high=True)
    s.SD2.acquire(2e4, ref='m2', threshold=50)

    # generate the sequence and upload it.
    my_seq = pulse_lib.mk_sequence([seg1, seg2, seg3])
    my_seq.n_rep = 10

    return my_seq


# create "AWG1","AWG2"
awgs, digs = init_hardware()

# create channels
pulse = init_pulselib(awgs, digs, virtual_gates=True)

t_measure = 5_000

# With current setup the angle of -0.228*PI results in a measurement with IQ angle is 0 (Q is 0.0)
pulse.set_digitizer_phase('SD2', -0.228*np.pi)

my_seq = create_seq(pulse)
#my_seq.set_hw_schedule(Hvi2ScheduleLoader(pulse, "SingleShot", digs[0]))
my_seq.set_hw_schedule(HardwareScheduleMock())
my_seq.set_acquisition(t_measure=t_measure)

param = my_seq.get_measurement_param('Test', upload='auto', iq_complex=False)
param.add_derived_param('m2_amp', lambda d:np.abs(d['m2_I']+1j*d['m2_Q']))
param.add_derived_param('m2_phase', lambda d:np.angle(d['m2_I']+1j*d['m2_Q']))

# Reading param uploads, plays and returns data
data = param()

data = param()

# plot_awgs(awgs+digs)

pt.figure()
for ch_name,values in zip(param.names, data):
    print(ch_name, values)
    if isinstance(values, (Sequence, np.ndarray)):
        if isinstance(values[0], complex):
            pt.plot(values.real, label=ch_name+' I')
            pt.plot(values.imag, label=ch_name+' Q')
        else:
            pt.plot(values, label=ch_name)
pt.legend()

