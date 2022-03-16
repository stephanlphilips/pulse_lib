from pulse_lib.base_pulse import pulselib
from pulse_lib.virtual_channel_constructors import virtual_gates_constructor
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
        from .init_keysight import awg1
        return [awg1], []
    if _backend == 'Keysight_QS':
        _ch_offset = 1
        from .init_keysight_qs import awg1
        return [awg1], []

pulse = None
def init_pulselib(awgs, digitizers, virtual_gates=False, bias_T_rc_time=None):
    global pulse

    pulse = pulselib(_backend)

    for awg in awgs:
        pulse.add_awg(awg)

    for dig in digitizers:
        pulse.add_digitizer(dig)

    awg1 = awgs[0].name
    # define channels
    pulse.define_channel('P1', awg1, 0 + _ch_offset)
    pulse.define_channel('P2', awg1, 1 + _ch_offset)

    # add limits on voltages for DC channel compensation (if no limit is specified, no compensation is performed).
    pulse.add_channel_compensation_limit('P1', (-200, 500))
    pulse.add_channel_compensation_limit('P2', (-200, 500))

    if bias_T_rc_time:
        pulse.add_channel_bias_T_compensation('P1', bias_T_rc_time)
        pulse.add_channel_bias_T_compensation('P2', bias_T_rc_time)

    dig_name = digitizers[0].name if len(digitizers) > 0 else 'Dig1'
    pulse.define_digitizer_channel('SD1', dig_name, 0 + _ch_offset)
    pulse.define_digitizer_channel('SD2', dig_name, 1 + _ch_offset)

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

    pulse.finish_init()

    return pulse

