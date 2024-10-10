
from pulse_lib.tests.configurations.test_configuration import context
#%%
import pulse_lib.segments.utility.looping as lp

from core_tools.sweeps.scans import Scan, sweep

import numpy as np


def test():
    pulse = context.init_pulselib(n_qubits=1)

    station = context.station

    # Pass LO frequency parameter to pulse-lib.
    # Pulse-lib calculates IQ frequency: f_IQ = f_MW - f_LO
    pulse.set_iq_lo('IQ1', station.mw_source.frequency)

    # Define q1 on IQ output IQ1 with qubit frequency 12.412e9
    pulse.define_qubit_channel('q1', 'IQ1', 12.412e9)

    # Set LO frequnency of vector source to 0.0.
    pulse.set_iq_lo('IQ1', 0.0)

    # Set qubit frequency to 0.0. (It may not differ more than 400MHz from IQ frequency)
    pulse.set_qubit_resonance_frequency('q1', 0.0)

    iq_freq = lp.linspace(100e6, 200e6, 101, name='iq_freq', unit='Hz', axis=0)
    iq_amplitude = 500 # mV

    s = pulse.mk_segment()

    # TODO Add init pulses

    # 1 us pulse
    s.q1.add_MW_pulse(0, 1000, iq_amplitude, iq_freq)

    # TODO Add readout pulses

    sequence = pulse.mk_sequence([s])
    sequence.n_rep = 100
    m_param = sequence.get_measurement_param()

    ds = Scan(
            sweep(station.mw_source.frequency, 2.0e9, 3.0e9, n_points=11, delay=0.01),
            sequence,
            m_param,
            ).run()

    ds = Scan(
            sweep(station.mw_source.frequency, np.arange(2.09, 3.0e9, 0.1e9), delay=0.01),
            sequence,
            m_param,
            ).run()


    context.plot_awgs(sequence)

    return None


if __name__ == '__main__':
    ds = test()
