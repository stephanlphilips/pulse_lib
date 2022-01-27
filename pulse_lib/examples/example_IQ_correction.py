import matplotlib.pyplot as pt

import qcodes as qc

from pulse_lib.tests.hw_schedule_mock import HardwareScheduleMock


from pulse_lib.tests.mock_m3202a import MockM3202A_fpga
from pulse_lib.tests.mock_m3202a_qs import MockM3202A_QS
from pulse_lib.base_pulse import pulselib
from pulse_lib.virtual_channel_constructors import IQ_channel_constructor

from utils.plot import plot_awgs

#import qcodes.logger as logger
from qcodes.logger import start_all_logging

start_all_logging()


if not qc.Station.default:
    station = qc.Station()
else:
    station = qc.Station.default

backend = 'Keysight'
#backend = 'Keysight_QS'
#backend = 'Qblox'
# There is not mock for Tektronix

def station_get_or_create(func):
    def wrapper(name, *args, **kwargs):
        try:
            return station[name]
        except:
            component = func(name, *args, **kwargs)
            station.add_component(component)
            return component
    return wrapper

@station_get_or_create
def add_M3202A_fpga(name, chassis, slot):
    return MockM3202A_fpga(name, chassis, slot)

@station_get_or_create
def add_M3202A_QS(name, chassis, slot):
    return MockM3202A_QS(name, chassis, slot)


def init_hardware():
    if backend == 'Keysight':
        awg1 = add_M3202A_fpga("AWG1", 0, 2)
    elif backend == 'Keysight_QS':
        awg1 = add_M3202A_QS("AWG1", 0, 2)
    else:
        raise NotImplementedError(f'No AWG for backend {backend}')
    return [awg1]

def init_pulselib(awgs):

    pulse = pulselib(backend=backend)

    for awg in awgs:
        pulse.add_awgs(awg.name, awg)

    # define channels
    pulse.define_channel('I1','AWG1', 1)
    pulse.define_channel('Q1','AWG1', 2)
    pulse.define_channel('I2','AWG1', 3)
    pulse.define_channel('Q2','AWG1', 4)
    pulse.add_channel_offset('I2', 10)
    pulse.add_channel_offset('Q2', -5)

    # define IQ output pair
    IQ_pair_1 = IQ_channel_constructor(pulse)
    IQ_pair_1.add_IQ_chan("I1", "I")
    IQ_pair_1.add_IQ_chan("Q1", "Q")
    # frequency of the MW source
    IQ_pair_1.set_LO(2.40e9)

    IQ_pair_2 = IQ_channel_constructor(pulse)
    IQ_pair_2.add_IQ_chan("I2", "I")
    IQ_pair_2.add_IQ_chan("Q2", "Q")
    # frequency of the MW source
    IQ_pair_2.set_LO(2.40e9)

    IQ_pair_1.add_virtual_IQ_channel("q0", 2.420e9)
    IQ_pair_2.add_virtual_IQ_channel("q1", 2.420e9,
                                     correction_gain=(0.9,1.0))
    IQ_pair_2.add_virtual_IQ_channel("q2", 2.420e9,
                                     correction_gain=(1.0,0.9))
    IQ_pair_2.add_virtual_IQ_channel("q3", 2.420e9,
                                     correction_phase=0.3)

    pulse.finish_init()

    return pulse

awgs = init_hardware()
p = init_pulselib(awgs)


f_drive = 2.420e9
t_X90 = 100
amplitude = 100
t_wait = 50

manip = p.mk_segment()
if backend == 'Keysight_QS':
    manip.q0.wait(t_wait-10)
    manip.reset_time()

manip.q0.add_MW_pulse(0, t_X90, amplitude, f_drive)
manip.q0.wait(t_wait)
manip.reset_time()
manip.q0.add_MW_pulse(0, t_X90, amplitude, f_drive)
manip.q1.add_MW_pulse(0, t_X90, amplitude, f_drive)
manip.q0.wait(t_wait)
manip.reset_time()
manip.q0.add_MW_pulse(0, t_X90, amplitude, f_drive)
manip.q2.add_MW_pulse(0, t_X90, amplitude, f_drive)
manip.q0.wait(t_wait)
manip.reset_time()
manip.q0.add_MW_pulse(0, t_X90, amplitude, f_drive)
manip.q3.add_MW_pulse(0, t_X90, amplitude, f_drive)

# generate the sequence from segments
my_seq = p.mk_sequence([manip])
my_seq.set_hw_schedule(HardwareScheduleMock())
my_seq.n_rep = 1
my_seq.sample_rate = 1e9

my_seq.upload()
my_seq.play()

plot_awgs(awgs)
pt.legend()
pt.grid(True)

