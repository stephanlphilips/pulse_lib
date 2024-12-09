
from pulse_lib.tests.configurations.test_configuration import context

#%%
import numpy as np
import matplotlib.pyplot as pt

from pulse_lib.qblox.pulsar_sequencers import PulsarConfig, VoltageSequenceBuilder

VoltageSequenceBuilder.verbose = True

def config_backend(pulse):
    if pulse._backend in ['Keysight', 'Keysight_QS']:
        context.station.AWG1.set_digital_filter_mode(3)
    if pulse._backend == 'Qblox':
        # increase resolution for fair check of algorithm
        PulsarConfig.NS_SUB_DIVISION = 20


def test_shuttle(
        t0=5.46,
        t_ramp=2.0,
        t_shuttle=13,
        offset_1=200.0,
        offset_2=150.0,
        amplitude_1=400,
        amplitude_2=400,
        phase_1=1.2,
        phase_2=1.7,
        frequency=-46.1e6,
        hres=True,
        ):
    pulse = context.init_pulselib(n_gates=2)
    config_backend(pulse)

    phase_1 *= np.pi
    phase_2 *= np.pi

    v1_start = offset_1 + amplitude_1 * np.sin(phase_1)
    v2_start = offset_2 + amplitude_2 * np.sin(phase_2)
    v1_stop = offset_1 + amplitude_1 * np.sin(phase_1 + 2*frequency*t_shuttle*1e-9*2*np.pi)
    v2_stop = offset_2 + amplitude_2 * np.sin(phase_2 + 2*frequency*t_shuttle*1e-9*2*np.pi)

    t_offset = 0e5

    s0 = pulse.mk_segment(hres=hres)
    s0.wait(0.050000002+t_offset)

    s = pulse.mk_segment(hres=hres)
    P1 = s.P1
    P2 = s.P2

    s.wait(t0, reset_time=True)
    P1.add_ramp_ss(0, t_ramp, 0, v1_start)
    P2.add_ramp_ss(0, t_ramp, 0, v2_start)
    s.reset_time()
    P1.add_block(0, t_shuttle, offset_1)
    P2.add_block(0, t_shuttle, offset_2)
    P1.add_sin(0, t_shuttle, amplitude_1, 2*frequency, phase_offset=phase_1)
    P1.add_sin(0, t_shuttle, 0.0, frequency, phase_offset=phase_1)
    P2.add_sin(0, t_shuttle, amplitude_2, 2*frequency, phase_offset=phase_2)
    P2.add_sin(0, t_shuttle, 0.0, frequency, phase_offset=phase_2)
    s.reset_time()
    P1.add_block(0, -1, v1_stop)
    P2.add_block(0, -1, v2_stop)
    s.wait(120, reset_time=True)

    sequence = pulse.mk_sequence([s0, s])
    sequence.n_rep = None

    context.plot_awgs(sequence,
                      ylim=(-1.10, 1.10),
                      xlim=(0+t_offset, 30+t_offset),
                      analogue_out=True,
                      create_figure=False,
                      )


#%%
if __name__ == '__main__':
    pt.close("all")
    for t in [4.0, 4.07, 4.33, 4.4, 4.5, 4.6, 4.7, 4.8,
              4.9, #4.95, 5.0,
              5.18, 5.91, 6.23, 7.02, 7.68,
              ]:
        pt.figure()
        test_shuttle(t0=t)

