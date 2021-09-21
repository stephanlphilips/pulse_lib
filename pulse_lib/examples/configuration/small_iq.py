from pulse_lib.tests.mock_m3202a import MockM3202A_fpga
from pulse_lib.base_pulse import pulselib
from pulse_lib.virtual_channel_constructors import virtual_gates_constructor, IQ_channel_constructor
import numpy as np
import qcodes


def init_hardware():
    # try to close all instruments of previous run
    try:
        qcodes.Instrument.close_all()
    except: pass

    awg1 = MockM3202A_fpga("AWG1", 0, 2)
    return [awg1]


def init_pulselib(awgs, virtual_gates=False, bias_T_rc_time=None):

    pulse = pulselib()

    for awg in awgs:
        pulse.add_awgs(awg.name, awg)

    # define channels
    pulse.define_channel('P1','AWG1', 1)
    pulse.define_channel('P2','AWG1', 2)
    pulse.define_channel('I1','AWG1', 3)
    pulse.define_channel('Q1','AWG1', 4)

    pulse.define_digitizer_channel('SD1', 'Digitizer', 1)

    # add limits on voltages for DC channel compenstation (if no limit is specified, no compensation is performed).
    pulse.add_channel_compensation_limit('P1', (-200, 500))
    pulse.add_channel_compensation_limit('P2', (-200, 500))

    if bias_T_rc_time:
        pulse.add_channel_bias_T_compensation('P1', bias_T_rc_time)
        pulse.add_channel_bias_T_compensation('P2', bias_T_rc_time)

    if virtual_gates:
        # set a virtual gate matrix
        virtual_gate_set_1 = virtual_gates_constructor(pulse)
        virtual_gate_set_1.add_real_gates('P1','P2')
        virtual_gate_set_1.add_virtual_gates('vP1','vP2')
        inv_matrix = 1.2*np.eye(2) - 0.1
        virtual_gate_set_1.add_virtual_gate_matrix(np.linalg.inv(inv_matrix))

    # define IQ output pair
    IQ_pair_1 = IQ_channel_constructor(pulse)
    IQ_pair_1.add_IQ_chan("I1", "I")
    IQ_pair_1.add_IQ_chan("Q1", "Q")
    # frequency of the MW source
    IQ_pair_1.set_LO(2.40e9)

    # add 1 qubit: q1
    IQ_pair_1.add_virtual_IQ_channel("q1")

    pulse.finish_init()

    return pulse

