import numpy as np
import logging
import matplotlib.pyplot as pt

import qcodes
import qcodes.logger as logger
from qcodes.logger import start_all_logging

from pulse_lib.base_pulse import pulselib
from pulse_lib.virtual_channel_constructors import IQ_channel_constructor
import pulse_lib.segments.utility.looping as lp

from pulse_lib.tests.mock_m3202a import MockM3202A
from pulse_lib.tests.hw_schedule_mock import HardwareScheduleMock



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
    p.define_channel('AWG_I', awg1.name, 3)
    p.define_channel('AWG_Q', awg1.name, 4)

    # NOTE: digitizer should be added when real hardware is used.
    p.define_digitizer_channel('SD1', 'Dig1', 1)

    #make virtual channels for IQ usage (also here, make one one of these object per MW source)
    IQ_pair_1 = IQ_channel_constructor(p)
    # set right association of the real channels with I/Q output.
    IQ_pair_1.add_IQ_chan("AWG_I", "I")
    IQ_pair_1.add_IQ_chan("AWG_Q", "Q")
    # frequency of the MW source
    IQ_pair_1.set_LO(3.20e9)

    IQ_pair_1.add_virtual_IQ_channel("q1")

    p.finish_init()

    return p


awg1 = MockM3202A("A1", 0, 2)

pulse = init_pulselib(awg1)

t_pulse = lp.linspace(100, 1000, 10, axis=0)
f_drive = lp.linspace(3.25e9, 3.30e9, 11, axis=1)
amplitude = 50

# init pulse
init = pulse.mk_segment()
init.P1.add_block(0, 100, 150)

manip = pulse.mk_segment()
manip.q1.add_MW_pulse(0, t_pulse, amplitude, f_drive)

# read-out
t_measure = 200 # short time for visibility of pulse
readout = pulse.mk_segment()
readout.P1.add_ramp_ss(0, 100, 0, 50)
readout.reset_time()
readout.P1.add_block(0, t_measure, 50)
readout.SD1.acquire(0, t_measure)

# generate the sequence from segments
my_seq = pulse.mk_sequence([init, manip, readout])
my_seq.set_hw_schedule(HardwareScheduleMock())
my_seq.n_rep = 1
my_seq.sample_rate = 1e9


for f in [0, 9]:
    for t in [0, 1, 9]:
        my_seq.upload([f,t])
        my_seq.play([f,t])
        pt.figure()
        pt.title(f'f={f_drive[f]/1e6:7.2f} MHz, t={t_pulse[t]} ns')
        awg1.plot()



