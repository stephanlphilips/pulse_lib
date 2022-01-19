import logging
from pprint import pprint
import numpy as np
import matplotlib.pyplot as pt

import qcodes.logger as logger
from qcodes.logger import start_all_logging

from pulse_lib.tests.hw_schedule_mock import HardwareScheduleMock

from configuration.medium_iq import init_hardware, init_pulselib

start_all_logging()
logger.get_file_handler().setLevel(logging.DEBUG)


def create_seq(pulse_lib):

    seg1 = pulse_lib.mk_segment(name='init')
    s = seg1
    s.P1.add_ramp_ss(0, 1000, -120, 150)
    s['P2'].add_block(0, 500, 50)
    s.vP3.add_block(600, 800, 100)

    seg2 = pulse_lib.mk_segment('manip')
    s = seg2
    s.vP4.add_ramp_ss(0, 100, 50, 100)
    s.vP4.add_ramp_ss(100, 200, 100, 50)
    s.P5.add_block(0, 200, -100)


    seg3 = pulse_lib.mk_segment('measure')
    s = seg3
    s.P1.add_block(0, 1e4, -90)
    s.P2.add_block(0, 1e4, 120)
    s.vP4.add_ramp_ss(1e4, 3e4, 0, 50)
    s.vP4.add_block(3e4, 4e4, 50)
    s.vP4.add_ramp_ss(4e4, 5e4, 50, 0)
    s.SD1.acquire(3.5e4)
    s.SD2.acquire(3.5e4)

    # generate the sequence and upload it.
    my_seq = pulse_lib.mk_sequence([seg1, seg2, seg3])
    my_seq.set_hw_schedule(HardwareScheduleMock())
    my_seq.n_rep = 4
    my_seq.sample_rate = 1e9

    return my_seq

def plot(seq, awgs):

    fig = pt.figure(1)
    fig.clear()

    for awg in awgs:
        awg.plot()

    pt.legend()
    pt.show()


# create "AWG1","AWG2"
awgs, digs = init_hardware()

# create channels
pulse = init_pulselib(awgs, digs, virtual_gates=True)

my_seq = create_seq(pulse)
t_measure = 10_000
my_seq.set_acquisition(t_measure=t_measure)

param = my_seq.get_acquisition_param('Test', upload='auto')
param.add_derived_param('Amp', lambda d:np.sqrt(d['SD1']**2 + d['SD2']**2))
param.add_derived_param('Diff', lambda d:d['SD1'] - d['SD2'])

# Reading param uploads, plays and returns data
data = param()

plot(my_seq, awgs+digs)

pt.figure()
names = param.names
for ch_name,values in zip(param.names, data):
    print(ch_name, values)
    pt.plot(values, label=ch_name)
pt.legend()

