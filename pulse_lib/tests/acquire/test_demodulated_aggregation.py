
from pulse_lib.tests.configurations.test_configuration import context

#%%

import numpy as np

class IQDemodulator:
    def __init__(self, digitizer, frequency):
        self.digitizer = digitizer
        self.frequency = frequency

    def demodulate(self, ch_data):
        sample_rate = self.digitizer.sample_rate.cache()
        n = ch_data.shape[-1]
        t = np.arange(0, n) * (1/sample_rate)
        demod_vector = np.exp(-2j*np.pi*self.frequency*t)
        return np.mean(ch_data * demod_vector, axis=-1)


def test_iqdemod_func():
    digitizer = context.station.Dig1
    sample_rate = 250e6
    digitizer.sample_rate(sample_rate)

    f = 10e6
    demod = IQDemodulator(digitizer, f).demodulate

    n = 250
    t = np.arange(0, n) * (1/sample_rate)
    data = np.cos(2*np.pi*f*t+np.pi/10)
    iq = demod(data)
    print(iq, np.abs(iq), np.angle(iq)/np.pi)


def test1():
    pulse = context.init_pulselib(n_gates=2, n_sensors=2, rf_sources=True)

    pulse.digitizer_channels['SD1'].iq_out=True

    # NOTE: assume M4i. Test does not yet work with other digitizers.
    digitizer = context.station.Dig1
    digitizer.sample_rate(250e6)
    # get actual sample rate from m4i.
    sample_rate = digitizer.sample_rate()

    f = 10e6
    t_measure = 100

    s = pulse.mk_segment()

    s.SD1.acquire(0, t_measure)
    s.wait(10000, reset_time=True)
    s.SD2.acquire(0, t_measure)
    s.wait(10000)

    sequence = pulse.mk_sequence([s])
    sequence.n_rep = 2
    sequence.set_acquisition(
            sample_rate=sample_rate,
            aggregate_func=IQDemodulator(digitizer, f).demodulate)

    m_param = sequence.get_measurement_param(iq_mode='amplitude+phase')
    context.add_hw_schedule(sequence)

    t = np.arange(0, t_measure, 1e9/sample_rate) # [ns]
    t *= 1e-9 # [s]

    # WORKAROUND for M4i: multiply amount of data by 2, because there are 2 triggers.
    context.set_mock_data({
            'SD1': [np.cos(2*np.pi*f*t+np.pi/10)]*2,
            'SD2': [0.6*np.exp(1j*(2*np.pi*f*t+np.pi/5))]*2,
            },
            repeat=sequence.n_rep,
            )

    return context.run('iq_demodulation', sequence, m_param)

def demodulate(data):
    # create vector with [-1,+1,-1,+1,...]
    n = data.shape[-1]
    v = np.full(n, -1.0)
    v = v.cumprod()
    if n%2 == 1:
        print('Warning: odd number of points')
        # set last to 0, because average of vector must be 0.
        v[-1] = 0
    return np.mean(data*v, axis=-1)

def test2():
    pulse = context.init_pulselib(n_gates=2, n_sensors=2, rf_sources=True)

    # NOTE: assume M4i. Test does not yet work with other digitizers.
    digitizer = context.station.Dig1
    digitizer.sample_rate(250e6)
    # get actual sample rate from m4i.
    sample_rate = digitizer.sample_rate()

    f = 10e6
    t_measure = 1000

    s = pulse.mk_segment()

    s.SD1.acquire(0, t_measure)
    s.wait(10000, reset_time=True)

    sequence = pulse.mk_sequence([s])
    sequence.n_rep = 2
    sequence.set_acquisition(
            sample_rate=f*2,
            aggregate_func=demodulate)

    m_param = sequence.get_measurement_param(iq_mode='amplitude+phase')
    context.add_hw_schedule(sequence)

    t = np.arange(0, t_measure, 1e9/sample_rate) # [ns]
    t *= 1e-9 # [s]

    context.set_mock_data({
            'SD1': [-np.sin(2*np.pi*f*t)],
            },
            repeat=sequence.n_rep,
            )

    return context.run('block_demodulation', sequence, m_param)


if __name__ == '__main__':
#    test_iqdemod_func()
#    ds1 = test1()
#    print(ds1.SD1_1_amp)
#    print(ds1.SD1_1_phase)
#    print(ds1.SD2_1_amp)
#    print(ds1.SD2_1_phase)
    ds2 = test2()
    print(ds2.SD1_1)
