import numpy as np
import logging
from pprint import pprint
import matplotlib.pyplot as pt

import qcodes.logger as logger
from qcodes.logger import start_all_logging

from pulse_lib.tests.hw_schedule_mock import HardwareScheduleMock

from configuration.medium_iq import init_hardware, init_pulselib

start_all_logging()
logger.get_file_handler().setLevel(logging.DEBUG)

def create_seq(pulse_lib):

    seg1 = pulse_lib.mk_segment(name='init', sample_rate=1e8)
    s = seg1
    s.P1.add_ramp_ss(0, 1000, -120, 150)
    s.P2.add_block(0, 500, 50)
    s.vP3.add_block(600, 800, 100)

    s.M1.add_marker(800, 900)

    # no sample rate specified: it uses default sample rate. In this example 2e8 Sa/s (see below).
    seg2 = pulse_lib.mk_segment('manip')
    s = seg2
    s.vP4.add_ramp_ss(0, 100, 50, 100)
    s.vP4.add_ramp_ss(100, 200, 100, 50)

    s.q1.add_MW_pulse(0, 300, 20, 2.435e9)
    s.q1.add_phase_shift(300, np.pi/2)

    seg2b = pulse_lib.mk_segment('manip2')
    s = seg2b
    s.vP4.add_ramp_ss(0, 100, 50, 100)
    s.vP4.add_ramp_ss(100, 200, 100, 50)

    s.q1.add_MW_pulse(0, 300, 20, 2.435e9)

    seg3 = pulse_lib.mk_segment('measure', 1e8)
    s = seg3
    s.P1.add_block(0, 1e5, 100)
    s.P2.add_block(0, 1e5, 120)
    s.vP4.add_block(1e4, 2e4, 120)
    s.reset_time()
    s.add_HVI_marker('ping', 99)

    # segment without data. Will be used for DC compensation with low sample rate
    seg4 = pulse_lib.mk_segment('dc compensation', 1e7)
    # wait 100 ns (i.e. 10 samples at 1e8 MSa/s)
    seg4.P1.wait(100)

    # generate the sequence and upload it.
    my_seq = pulse_lib.mk_sequence([seg1, seg2, seg2b, seg3, seg4])
    my_seq.set_hw_schedule(HardwareScheduleMock())
    my_seq.n_rep = 1

    return my_seq

def plot(seq, job, awgs):
#    uploader = seq.uploader
    print(f'sequence: {seq.shape}')
    print(f'job index:{job.index}, sequences:{len(job.sequence)}')
    print(f'  sample_rate:{job.default_sample_rate} playback_time:{job.playback_time}')

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

logging.info(f'sequence shape: {my_seq.shape}')

job = my_seq.upload([0])

my_seq.play([0], release=False)

plot(my_seq, job, awgs)
pprint(job.upload_info)

my_seq.play([0], release=True)
my_seq.uploader.release_jobs()

