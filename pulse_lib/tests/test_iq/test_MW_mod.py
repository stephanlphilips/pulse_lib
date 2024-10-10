
from pulse_lib.tests.configurations.test_configuration import context

#%%
import numpy as np

from pulse_lib.segments.utility import looping as lp

def get_AM_envelope(delta_t: float, sample_rate: float):
    npt = int(delta_t*sample_rate + 0.5)
    return np.linspace(0, 1.0, npt)


def get_AM_envelope2(delta_t: float, sample_rate: float, alpha: float):
    npt = int(delta_t*sample_rate + 0.5)
    return np.linspace(alpha, 1.0, npt)



def test1():
    pulse = context.init_pulselib(n_gates=1, n_qubits=1)

    f_q1 = pulse.qubit_channels['q1'].resonance_frequency

    s = pulse.mk_segment()

    s.q1.add_MW_pulse(0, 200, 100, f_q1, AM=get_AM_envelope)

    sequence = pulse.mk_sequence([s])
    sequence.n_rep = 1
    context.plot_awgs(sequence)

    return None


def test2():
    pulse = context.init_pulselib(n_gates=1, n_qubits=1)

    f_q1 = pulse.qubit_channels['q1'].resonance_frequency
    alpha = lp.linspace(0.2, 1.0, 5, "alpha", axis=0)

    s = pulse.mk_segment()

    s.q1.add_MW_pulse(0, 200, 100, f_q1, AM=get_AM_envelope2, alpha=alpha)

    sequence = pulse.mk_sequence([s])
    sequence.n_rep = 1
    for i in range(len(alpha)):
        context.plot_awgs(sequence, index=(i,))

    return None


class IQ_modulation:
    def __init__(
            self,
            I: list[lp.loop_obj] | list[float],
            Q: list[lp.loop_obj] | list[float],
            segment_length: int):
        self.I = I
        self.Q = Q
        self.segment_length = segment_length
        if len(I) != len(Q):
            raise Exception("I and Q list must have same length")
        self.n_segments = len(I)
        self.length = self.n_segments * segment_length

        self.indexed = hasattr(I[0], 'axis')
        if self.indexed:
            axis = I[0].axis[0]
            n = I[0].shape[0]
            for loop in I + Q:
                if loop.axis != [axis] or loop.shape != (n,):
                    raise Exception(f"All loops must have axis {axis} and shape {(n,)}. "
                                    f"Error on loop {loop}")
            self.loop = lp.arange(0, n, name="index", axis=axis)
        else:
            self.loop = None

    def get_index_loop(self):
        return self.loop

    def _render_IQ(self, delta_t: float, sample_rate: float, index: float = 0.0):
        # NOTE: looping arguments are always float. Cast to int.
        index = int(index)
        if round(delta_t*sample_rate) != self.length:
            raise Exception("Duration of wave doesn't match modulation spec. "
                            f"Expected {self.length}, but got {round(delta_t*sample_rate)} ns")
        if self.indexed:
            i_values = [i[index] for i in self.I]
            q_values = [q[index] for q in self.Q]
        else:
            i_values = self.I
            q_values = self.Q

        iq = np.zeros(self.length, dtype=complex)
        size = self.segment_length
        for i in range(self.n_segments):
            iq[i*size : (i+1)*size] = i_values[i] + 1j*q_values[i]

        return iq

    def get_AM_envelope_indexed(self, delta_t: float, sample_rate: float, index: float = 0.0):
        iq_data = self._render_IQ(delta_t, sample_rate, index)
        return np.abs(iq_data)

    def get_PM_envelope_indexed(self, delta_t: float, sample_rate: float, index: float = 0.0):
        iq_data = self._render_IQ(delta_t, sample_rate, index)
        return np.angle(iq_data)


def testIQloop():
    pulse = context.init_pulselib(n_gates=1, n_qubits=1)

    f_q1 = pulse.qubit_channels['q1'].resonance_frequency

    iq_mod = IQ_modulation(
        I=[
            lp.array([0.5, 0.2, 0.1], "i1", axis=0),
            lp.array([0.5, 0.8, -0.1], "i2", axis=0),
        ],
        Q=[
            lp.array([0.0, 0.2, 0.5], "q1", axis=0),
            lp.array([0.0, 0.8, -0.5], "q2", axis=0),
        ],
        segment_length=100,
        )

    s = pulse.mk_segment()

    s.q1.add_MW_pulse(0, 200, 100, f_q1,
                      AM=iq_mod.get_AM_envelope_indexed,
                      PM=iq_mod.get_PM_envelope_indexed,
                      index=iq_mod.get_index_loop())

    sequence = pulse.mk_sequence([s])
    sequence.n_rep = 1
    for i in range(len(iq_mod.get_index_loop())):
        context.plot_awgs(sequence, index=(i,))

    return None



#%%

if __name__ == '__main__':
    # ds1 = test1()
    # ds2 = test2()
    testIQloop()
