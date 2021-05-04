import numpy as np
import time
import logging
from pprint import pprint
import matplotlib.pyplot as pt

import qcodes
import qcodes.logger as logger
from qcodes.logger import start_all_logging

from pulse_lib.base_pulse import pulselib
from pulse_lib.tests.mock_m3202a import MockM3202A, MockM3202A_fpga
from pulse_lib.tests.hw_schedule_mock import HardwareScheduleMock

import pulse_lib.segments.utility.looping as lp
from pulse_lib.virtual_channel_constructors import IQ_channel_constructor, virtual_gates_constructor

import scipy.signal.windows as windows

start_all_logging()
logger.get_file_handler().setLevel(logging.DEBUG)


try:
    qcodes.Instrument.close_all()
except: pass


def set_channel_props(pulse, channel_name, compensation_limits=(0,0), attenuation=1.0, delay=0):
    pulse.add_channel_compensation_limit(channel_name, compensation_limits)
    pulse.awg_channels[channel_name].attenuation = attenuation
    pulse.add_channel_delay(channel_name, delay)



def init_pulselib(awg1, awg2, awg3):
    p = pulselib()
    p.add_awg(awg1)
    p.add_awg(awg2)
    p.add_awg(awg3)
    p.define_channel('P1', awg1.name, 1)
    p.define_channel('P2', awg1.name, 2)
    p.define_channel('P3', awg1.name, 3)
    p.define_channel('P4', awg1.name, 4)

    p.define_marker('M1', awg3.name, 3, setup_ns=40, hold_ns=20)
    p.define_marker('M2', awg3.name, 0, setup_ns=40, hold_ns=20)

    set_channel_props(p, 'P1', compensation_limits=(-100,100), attenuation=2.0, delay=0)
    set_channel_props(p, 'P2', compensation_limits=(-50,50), attenuation=1.0, delay=0)
    set_channel_props(p, 'P3', compensation_limits=(-80,80), attenuation=1.0, delay=0)

    p.define_channel('I1', awg2.name, 1)
    p.define_channel('Q1', awg2.name, 2)
    p.define_channel('I2', awg2.name, 3)
    p.define_channel('Q2', awg2.name, 4)

    set_channel_props(p, 'I1', compensation_limits=(-80,80), attenuation=1.0, delay=0)

    p.finish_init()

    # add virtual channels.

    virtual_gate_set_1 = virtual_gates_constructor(p)
    virtual_gate_set_1.add_real_gates('P1','P2','P3','P4')
    virtual_gate_set_1.add_virtual_gates('vP1','vP2','vP3','vP4')
    virtual_gate_set_1.add_virtual_gate_matrix(0.9*np.eye(4) + 0.1)

    #make virtual channels for IQ usage (also here, make one one of these object per MW source)
    IQ_chan_set_1 = IQ_channel_constructor(p)
    # set right association of the real channels with I/Q output.
    IQ_chan_set_1.add_IQ_chan("I1", "I")
    IQ_chan_set_1.add_IQ_chan("Q1", "Q")
    IQ_chan_set_1.add_marker("M1")
    # frequency of the MW source
    IQ_chan_set_1.set_LO(2e9)
    IQ_chan_set_1.add_virtual_IQ_channel("MW_qubit_1")
    IQ_chan_set_1.add_virtual_IQ_channel("MW_qubit_2")

    IQ_chan_set_2 = IQ_channel_constructor(p)
    # set right association of the real channels with I/Q output.
    IQ_chan_set_2.add_IQ_chan("I2", "I")
    IQ_chan_set_2.add_IQ_chan("Q2", "Q")
    IQ_chan_set_2.add_marker("M2")
    # frequency of the MW source
    IQ_chan_set_2.set_LO(2e9)
    IQ_chan_set_2.add_virtual_IQ_channel("MW_qubit_3")
    IQ_chan_set_2.add_virtual_IQ_channel("MW_qubit_4")

    return p


def tukey(duration, sample_rate):
    n_points = int(duration * sample_rate)
    return windows.tukey(n_points, alpha=0.5)

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

    s.MW_qubit_2.add_MW_pulse(50, 150, 50, 200e6, AM=tukey)

    s.MW_qubit_1.add_MW_pulse(250, 300, 60, 50e6)

    s.MW_qubit_3.add_MW_pulse(250, 300, 60, 50e6)

    seg3 = pulse_lib.mk_segment('measure', 1e8)
    s = seg3
    s.P1.add_block(0, 1e5, 100)
    s.P2.add_block(0, 1e5, 120)
    s.vP4.add_block(1e4, 9e4, 120)
    s.reset_time()
    s.add_HVI_marker('ping', 99)


    # segment without data. Will be used for DC compensation with low sample rate
    seg4 = pulse_lib.mk_segment('dc compensation', 1e7)
    # wait 10 ns (i.e. 1 sample at 1e8 MSa/s)
    seg4.P1.wait(100)

    # generate the sequence and upload it.
    my_seq = pulse_lib.mk_sequence([seg1, seg2, seg3, seg4])
    my_seq.set_hw_schedule(HardwareScheduleMock())
    my_seq.n_rep = 1
#    my_seq.sample_rate = 2e8

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

index = 0
def play_next():
    global index
    index += 1
    job = my_seq.upload([index])
    plot(my_seq, job)
    print(job.upload_info)
    my_seq.play([index], release=True)


awg1 = MockM3202A("A1", 0, 2)
awg2 = MockM3202A("A2", 0, 3)
awg3 = MockM3202A_fpga("A3", 0, 4)

pulse = init_pulselib(awg1, awg2, awg3)

my_seq = create_seq(pulse)

logging.info(f'sequence shape: {my_seq.shape}')

job = my_seq.upload()

my_seq.play(release=False)

plot(my_seq, job, (awg1, awg2, awg3) )
pprint(job.upload_info)

my_seq.play(release=True)
my_seq.uploader.release_jobs()

