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
    IQ_pair_1.set_LO(3.200e9)

    IQ_pair_1.add_virtual_IQ_channel("q1")

    p.finish_init()

    return p


awg1 = MockM3202A("A1", 0, 2)

pulse = init_pulselib(awg1)

gates = ['P1','P2']
v_init = [70, 20]
v_read = [30, 25]
t_measure = 100 # short time for visibility of other pulses

f_drive = 3.220e9
t_X90 = 50
amplitude = 50

t_wait = lp.linspace(0, 200, 21, axis=0)

# init pulse
init = pulse.mk_segment()
init.add_block(0, 100, gates, v_init)
init.add_ramp(100, 130, gates, v_init, [0,0])

# Ramsey
manip = pulse.mk_segment()
manip.q1.add_MW_pulse(0, t_X90, amplitude, f_drive)
manip.q1.wait(t_wait)
manip.q1.reset_time()
manip.q1.add_MW_pulse(0, t_X90, amplitude, f_drive)

# read-out
readout = pulse.mk_segment()
readout.add_ramp(0, 100, gates, [0,0], v_read)
readout.reset_time()
readout.add_block(0, t_measure, gates, v_read)
readout.SD1.acquire(0, t_measure)

# generate the sequence from segments
my_seq = pulse.mk_sequence([init, manip, readout])
my_seq.set_hw_schedule(HardwareScheduleMock())
my_seq.n_rep = 1
my_seq.sample_rate = 1e9


for t in [0, 1, 2, 3, 10]:
    my_seq.upload([t])
    my_seq.play([t])
    pt.figure()
    pt.title(f't={t_wait[t]} ns')

    # plot reference sine
#    t = np.arange(-130, 2*t_X90+t_wait[t]+100+t_measure)
#    s = np.cos(2*np.pi*20e6*t*1e-9)*0.050
#    pt.plot(t+130, s, ':', color='gray', label='ref')

    awg1.plot()
    pt.legend()

