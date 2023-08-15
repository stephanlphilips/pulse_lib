
from pulse_lib.tests.configurations.test_configuration import context

#%%
from core_tools.GUI.keysight_videomaps.data_getter.scan_generator_Qblox import (
        construct_1D_scan_fast, construct_2D_scan_fast)


def test1D():
    pulse = context.init_pulselib(n_gates=4, n_sensors=2, n_markers=1)
    pulse.add_channel_delay('SD1', 165)
    m_param = construct_1D_scan_fast(
            'P1', 100, 200,
            10_000, False, pulse,
            channels=['SD1'],
            enabled_markers=['M1'],
            acquisition_delay_ns=500,
            line_margin=1)

    return m_param, m_param()


def test2D():
    pulse = context.init_pulselib(n_gates=4, n_sensors=2, n_qubits=2, n_markers=1,
                                  rf_sources=True, virtual_gates=True)
    pulse.add_channel_delay('SD2', 168)
    m_param = construct_2D_scan_fast(
            'vP1', 40/0.145, 200,
            'vP2', 40/0.145, 200,
            12_000, True, pulse,
            channels=['SD1', 'SD2'],
#            enabled_markers=['M1'],
            acquisition_delay_ns=500,
            line_margin=1)

    return m_param, m_param()


#%%
if __name__ == '__main__':
    param, data = test1D()
    param, data = test2D()

