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
        from .init_pulsars import qcm0, qrm1
        return [qcm0], [qrm1]
    if _backend == 'Keysight':
        _ch_offset = 1
        from .init_keysight import awg1, dig1
        return [awg1], [dig1]
    if _backend == 'Keysight_QS':
        _ch_offset = 1
        from .init_keysight_qs import awg1, dig1
        return [awg1], [dig1]


def init_pulselib(awgs, digitizers, virtual_gates=False, bias_T_rc_time=None,
                  lo_frequency=None):

    pulse = pulselib(_backend)
    pulse.configure_digitizer = True

    for awg in awgs:
        pulse.add_awg(awg)

    for dig in digitizers:
        pulse.add_digitizer(dig)

    awg1 = awgs[0].name
    # define channels
    pulse.define_channel('P1', awg1, 0 + _ch_offset)
    pulse.define_channel('P2', awg1, 1 + _ch_offset)
    pulse.define_channel('I1', awg1, 2 + _ch_offset)
    pulse.define_channel('Q1', awg1, 3 + _ch_offset)

    dig_name = digitizers[0].name if len(digitizers) > 0 else 'Dig1'
    pulse.define_digitizer_channel('SD1', dig_name, 1)

    # add limits on voltages for DC channel compensation (if no limit is specified, no compensation is performed).
    pulse.add_channel_compensation_limit('P1', (-200, 500))
    pulse.add_channel_compensation_limit('P2', (-200, 500))

    if bias_T_rc_time:
        pulse.add_channel_bias_T_compensation('P1', bias_T_rc_time)
        pulse.add_channel_bias_T_compensation('P2', bias_T_rc_time)

    if virtual_gates:
        # set a virtual gate matrix
        pulse.add_virtual_matrix(
                name='virtual-gates',
                real_gate_names=['P1','P2'],
                virtual_gate_names=['vP1','vP2'],
                matrix=[
                    [+1.0, -0.1],
                    [-0.1, +1.0],
                    ]
                )

    q1_resonant_frequency = 2.430e9
    q2_resonant_frequency = 2.450e9

    # frequency of the MW source
    lo_freq = lo_frequency if lo_frequency is not None else 2.400e9

    # combine outputs I1 and Q1 in IQ pair
    IQ_pair_1 = IQ_channel_constructor(pulse)
    IQ_pair_1.add_IQ_chan("I1", "I")
    IQ_pair_1.add_IQ_chan("Q1", "Q")
    # set frequency of LO: this can be the qcodes parameter of the VSG
    IQ_pair_1.set_LO(lo_freq)

    # add qubits: q1 and q2
    IQ_pair_1.add_virtual_IQ_channel("q1", q1_resonant_frequency)
    IQ_pair_1.add_virtual_IQ_channel("q2", q2_resonant_frequency)

    pulse.finish_init()

    return pulse

