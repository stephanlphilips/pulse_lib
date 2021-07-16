import numpy as np
import logging
from pprint import pprint
import matplotlib.pyplot as pt

import qcodes
import qcodes.logger as logger
from qcodes.logger import start_all_logging

from pulse_lib.base_pulse import pulselib
from pulse_lib.tests.mock_m3202a import MockM3202A
from pulse_lib.tests.hw_schedule_mock import HardwareScheduleMock

from pulse_lib.virtual_channel_constructors import virtual_gates_constructor


start_all_logging()
logger.get_file_handler().setLevel(logging.DEBUG)


try:
    qcodes.Instrument.close_all()
except: pass


def set_channel_props(pulse, channel_name, compensation_limits=(0,0), attenuation=1.0, delay=0):
    pulse.add_channel_compensation_limit(channel_name, compensation_limits)
    pulse.awg_channels[channel_name].attenuation = attenuation
    pulse.add_channel_delay(channel_name, delay)



def init_pulselib(awg1):
    p = pulselib()
    p.add_awg(awg1)

    p.define_channel('P1', awg1.name, 1)
    p.define_channel('P2', awg1.name, 2)
    p.define_channel('P3', awg1.name, 3)
    p.define_channel('P4', awg1.name, 4)

    set_channel_props(p, 'P1', compensation_limits=(-100,100), attenuation=1.0, delay=0)
    set_channel_props(p, 'P2', compensation_limits=(-50,50), attenuation=1.0, delay=0)
    set_channel_props(p, 'P3', compensation_limits=(-80,80), attenuation=0.5, delay=0)

    p.finish_init()

    # add virtual channels.
    virtual_gate_set_1 = virtual_gates_constructor(p)
    virtual_gate_set_1.add_real_gates('P1','P2','P3','P4')
    virtual_gate_set_1.add_virtual_gates('vP1','vP2','vP3','vP4')
    # Virtual matrix with 1.0 on diagonal and 0.1 elsewhere
    virtual_gate_set_1.add_virtual_gate_matrix(0.9*np.eye(4) + 0.1)

    return p


def create_seq(pulse_lib):

    seg1 = pulse_lib.mk_segment()

    s = seg1
    s.P1.add_ramp_ss(0, 400, 0, 150)
    s.P1.add_block(400, 600, 150)
    s.P1.add_ramp_ss(600, 1000, 150, 50)
    s.P2.add_block(1000, 1500, 50)
    s.P2.wait(500)

    seg2 = pulse_lib.mk_segment()
    s = seg2
    s.vP1.add_ramp_ss(0, 400, 0, 150)
    s.vP1.add_block(400, 600, 150)
    s.vP1.add_ramp_ss(600, 1000, 150, 50)
    s.vP2.add_block(1000, 1500, 50)
    s.vP2.wait(500)

    # generate the sequence from segments
    my_seq = pulse_lib.mk_sequence([seg1, seg2])
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


awg1 = MockM3202A("A1", 0, 2)

pulse = init_pulselib(awg1)

my_seq = create_seq(pulse)

job = my_seq.upload()

my_seq.play()

plot(my_seq, job, [awg1] )
pprint(job.upload_info)


