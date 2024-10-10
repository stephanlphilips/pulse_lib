import logging
from pprint import pprint
import numpy as np
import matplotlib.pyplot as pt

import qcodes.logger as logger
from qcodes.logger import start_all_logging

from pulse_lib.tests.hw_schedule_mock import HardwareScheduleMock

from configuration.medium import init_hardware, init_pulselib
from utils.plot import plot_awgs

#start_all_logging()
#logger.get_file_handler().setLevel(logging.DEBUG)


def create_seq(pulse_lib):

    seg1 = pulse_lib.mk_segment(name='init')
    s = seg1
    s.vP1.add_block(0, 500, 50)

    seg2 = pulse_lib.mk_segment('manip')
    s = seg2
    s.vP2.add_ramp_ss(0, 100, 50, 100)
    s.vP2.add_ramp_ss(100, 200, 100, 50)

    seg3 = pulse_lib.mk_segment('measure')
    s = seg3
    s.vP1.add_block(0, 1e4, -90)
    s.vP2.add_block(0, 1e4, 120)
    s.vP3.add_ramp_ss(1e4, 2e4, 0, 50)
    s.vP3.add_block(2e4, 3e4, 50)
    s.vP3.add_ramp_ss(3e4, 3.5e4, 50, 0)
    s.vP1.add_block(2e4, 3e4, -100)
    s.vP2.add_block(2e4, 3e4, 120)
    s.SD1.acquire(2e4)
    s.SD2.acquire(2e4)

    # generate the sequence and upload it.
    my_seq = pulse_lib.mk_sequence([seg1, seg2, seg3])
    my_seq.n_rep = 10
    my_seq.sample_rate = 1e9

    return my_seq


# create "AWG1","AWG2", "DIG1"
awgs, digs = init_hardware()

# create channels
pulse = init_pulselib(awgs, digs, virtual_gates=True)

# With current setup the angle of -0.228*PI results in a measurement with IQ angle is 0 (Q is 0.0)
pulse.set_digitizer_phase('SD2', -0.228*np.pi)

my_seq = create_seq(pulse)
my_seq.set_acquisition(t_measure=1e4)
# with Downsampling:
# my_seq.set_acquisition(t_measure=1e4, sample_rate=0.5e6)

my_seq.set_hw_schedule(HardwareScheduleMock())
# my_seq.set_hw_schedule(Hvi2ScheduleLoader(pulse, "SingleShot", digs[0]))

logging.info(f'sequence shape: {my_seq.shape}')

job = my_seq.upload()

my_seq.play(release=False)

data = my_seq.get_measurement_results(iq_complex=False)

#plot_awgs(awgs+digs)

pt.figure()
for ch_name,values in data.items():
    if isinstance(values[0], complex):
        pt.plot(np.real(values), label=ch_name+' I')
        pt.plot(np.imag(values), label=ch_name+' Q')
    else:
        pt.plot(values, label=ch_name)
pt.legend()
#pprint(job.upload_info)

my_seq.play(release=True)
my_seq.uploader.release_jobs()

