
from pulse_lib.tests.configurations.test_configuration import context

#%%
import numpy as np
import matplotlib.pyplot as pt

from pulse_lib.qblox.pulsar_sequencers import PulsarConfig, Voltage1nsSequenceBuilder
from pulse_lib.qblox.pulsar_uploader import UploadAggregator

UploadAggregator.verbose = True
Voltage1nsSequenceBuilder.verbose = True

def config_backend(pulse):
    if pulse._backend in ['Keysight', 'Keysight_QS']:
        context.station.AWG1.set_digital_filter_mode(3)


def test1(t1, t2=10, n=20, hres=True):
    """
    Special test for Qblox to see whether many repeated unaligned sines
    still results in a limited number of waveforms and limited
    waveform memory use.
    """

    pulse = context.init_pulselib(n_gates=1)
    config_backend(pulse)

    print("length", n*t2*2)

    s = pulse.mk_segment(hres=hres)
    P1 = s.P1

    P1.add_block(0, t1, 1000)
    P1.reset_time()
    for _ in range(n):
        P1.add_sin(0, t2, 1000, 60e6, phase_offset=0)
        P1.reset_time()
        P1.add_sin(0, t2, 1000, -60e6, phase_offset=t2*60e6*1e-9*2*np.pi)
        P1.reset_time()

    s.wait(10)

    sequence = pulse.mk_sequence([s])
    sequence.n_rep = None

    context.plot_awgs(sequence,
                      ylim=(-1.10, 1.10),
                      # xlim=(5, 30),
                      # analogue_out=True,
                      # analogue_shift=4.0-t1,
                      )



#%%
if __name__ == '__main__':
    test1(10.4, t2=16.7, n=500)
