from pulse_lib.base_pulse import pulselib
from pulse_lib.virtual_channel_constructors import virtual_gates_constructor, IQ_channel_constructor
import numpy as np

_backend = 'Qblox'
#_backend = 'Keysight'
#_backend = 'Keysight_QS'

_ch_offset = 0

def init_hardware():
    global _ch_offset

    if _backend == 'Qblox':
        _ch_offset = 0
        from .init_pulsars_3 import qcm0, qcm2, qrm1
        return [qcm0, qcm2], [qrm1]
    if _backend == 'Keysight':
        _ch_offset = 1
        from .init_keysight import awg1, awg2
        return [awg1,awg2], []
    if _backend == 'Keysight_QS':
        _ch_offset = 1
        from .init_keysight_qs import awg1, dig1
        TODO()
        return [awg1], [dig1]


def init_pulselib(awgs, digitizers, virtual_gates=False, bias_T_rc_time=None):

    pulse = pulselib(_backend)

    for awg in awgs:
        pulse.add_awg(awg)

    for dig in digitizers:
        pulse.add_digitizer(dig)

    awg1 = awgs[0].name
    awg2 = awgs[1].name
    # define channels
    pulse.define_channel('P1', awg1, 0 + _ch_offset)
    pulse.define_channel('P2', awg1, 1 + _ch_offset)
    pulse.define_channel('I1', awg1, 2 + _ch_offset)
    pulse.define_channel('Q1', awg1, 3 + _ch_offset)

    pulse.define_channel('P3', awg2, 0 + _ch_offset)
    pulse.define_channel('P4', awg2, 1 + _ch_offset)
    pulse.define_channel('I2', awg2, 2 + _ch_offset)
    pulse.define_channel('Q2', awg2, 3 + _ch_offset)

    pulse.define_marker('M1', awg1, 0, setup_ns=40, hold_ns=20)
    pulse.define_marker('M2', awg2, 2, setup_ns=40, hold_ns=20)

    dig_name = digitizers[0].name if len(digitizers) > 0 else 'Dig1'
    pulse.define_digitizer_channel('SD1', dig_name, 1)

    # add limits on voltages for DC channel compensation (if no limit is specified, no compensation is performed).
    pulse.add_channel_compensation_limit('P1', (-100, 100))
    pulse.add_channel_compensation_limit('P2', (-50, 50))
    pulse.add_channel_compensation_limit('P3', (-80, 80))

    pulse.awg_channels['P1'].attenuation = 0.5

    if bias_T_rc_time:
        pulse.add_channel_bias_T_compensation('P1', bias_T_rc_time)
        pulse.add_channel_bias_T_compensation('P2', bias_T_rc_time)

    if virtual_gates:
        # set a virtual gate matrix
        virtual_gate_set_1 = virtual_gates_constructor(pulse)
        virtual_gate_set_1.add_real_gates('P1','P2', 'P3', 'P4')
        virtual_gate_set_1.add_virtual_gates('vP1','vP2', 'vP3', 'vP4')
        inv_matrix = 1.2*np.eye(4) - 0.05
        virtual_gate_set_1.add_virtual_gate_matrix(np.linalg.inv(inv_matrix))

    # define IQ output pair
    IQ_pair_1 = IQ_channel_constructor(pulse)
    IQ_pair_1.add_IQ_chan("I1", "I")
    IQ_pair_1.add_IQ_chan("Q1", "Q")
    IQ_pair_1.add_marker("M1")
    # frequency of the MW source
    IQ_pair_1.set_LO(2.400e9)

    # add 2 qubits: q2
    IQ_pair_1.add_virtual_IQ_channel("q1", 2.435e9)
    IQ_pair_1.add_virtual_IQ_channel("q2", 2.450e9)

    # define IQ output pair
    IQ_pair_2 = IQ_channel_constructor(pulse)
    IQ_pair_2.add_IQ_chan("I2", "I")
    IQ_pair_2.add_IQ_chan("Q2", "Q")
    IQ_pair_2.add_marker("M2")
    # frequency of the MW source
    IQ_pair_2.set_LO(2.800e9)

    # add qubits:
    IQ_pair_2.add_virtual_IQ_channel("q3", 2.900e9)
    IQ_pair_2.add_virtual_IQ_channel("q4", 2.700e9)

    pulse.finish_init()

    return pulse

