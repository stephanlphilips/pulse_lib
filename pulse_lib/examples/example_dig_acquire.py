import numpy as np
import time
import logging
from pprint import pprint
import matplotlib.pyplot as pt

import qcodes
import qcodes.logger as logger
from qcodes.logger import start_all_logging

from pulse_lib.base_pulse import pulselib
from pulse_lib.tests.mock_m3202a import MockM3202A_fpga
from pulse_lib.tests.mock_m3202a_qs import MockM3202A_QS
from pulse_lib.tests.mock_m3102a_qs import MockM3102A_QS
from pulse_lib.tests.hw_schedule_mock import HardwareScheduleMock

import pulse_lib.segments.utility.looping as lp
from pulse_lib.virtual_channel_constructors import IQ_channel_constructor, virtual_gates_constructor

import scipy.signal.windows as windows

start_all_logging()
logger.get_file_handler().setLevel(logging.DEBUG)


try:
    qcodes.Instrument.close_all()
except: pass


def set_channel_props(pulse, channel_name, compensation_limits=(0,0), attenuation=1.0, delay=0, bias_T_rc_time=None):
    pulse.add_channel_compensation_limit(channel_name, compensation_limits)
    pulse.add_channel_attenuation(channel_name, attenuation)
    pulse.add_channel_delay(channel_name, delay)
    pulse.add_channel_bias_T_compensation(channel_name, bias_T_rc_time)


def init_pulselib(awgs, digitizer):
    p = pulselib(backend='Keysight_QS')
    for awg in awgs:
        p.add_awg(awg)
    p.add_digitizer(digitizer)

    awg1, awg2 = awgs

    p.define_channel('P1', awg1.name, 1)
    p.define_channel('P2', awg1.name, 2)
    p.define_channel('P3', awg1.name, 3)
    p.define_channel('P4', awg1.name, 4)

    set_channel_props(p, 'P1', compensation_limits=(-500,500), attenuation=2.0, delay=0, bias_T_rc_time=0.001)
    set_channel_props(p, 'P2', compensation_limits=(-250,250), attenuation=1.0, delay=0, bias_T_rc_time=0.001)
    set_channel_props(p, 'P3', compensation_limits=(-280,280), attenuation=1.0, delay=0, bias_T_rc_time=0.001)
    set_channel_props(p, 'P4', bias_T_rc_time=0.001)

    p.define_digitizer_channel('SE1', digitizer.name, 1)

    p.finish_init()

    # add virtual channels.

    virtual_gate_set_1 = virtual_gates_constructor(p)
    virtual_gate_set_1.add_real_gates('P1','P2','P3','P4')
    virtual_gate_set_1.add_virtual_gates('vP1','vP2','vP3','vP4')
    virtual_gate_set_1.add_virtual_gate_matrix(0.9*np.eye(4) + 0.1)

    return p


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


    seg3 = pulse_lib.mk_segment('measure')
    s = seg3
    s.P1.add_block(0, 1e5, -90)
    s.P2.add_block(0, 1e5, 120)
    s.vP4.add_ramp_ss(1e4, 3e4, 0, 50)
    s.vP4.add_block(3e4, 7e4, 50)
    s.vP4.add_ramp_ss(7e4, 9e4, 50, 0)
    s.SE1.acquire(4e4, t_measure=2e4)

    # generate the sequence and upload it.
    my_seq = pulse_lib.mk_sequence([seg1, seg2, seg3])
    my_seq.set_hw_schedule(HardwareScheduleMock())
    my_seq.n_rep = 1
    my_seq.sample_rate = 1e9

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


awg1 = MockM3202A_fpga("A1", 0, 2)
awg2 = MockM3202A_QS("A2", 0, 4)
dig = MockM3102A_QS("Dig", 0, 9)

pulse = init_pulselib([awg1, awg2], dig)

my_seq = create_seq(pulse)

logging.info(f'sequence shape: {my_seq.shape}')

job = my_seq.upload([0])

my_seq.play([0], release=False)

plot(my_seq, job, (awg1,dig) )
pprint(job.upload_info)

my_seq.play([0], release=True)
my_seq.uploader.release_jobs()

