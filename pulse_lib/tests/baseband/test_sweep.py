
from pulse_lib.tests.configurations.test_configuration import context


#%%
import pulse_lib.segments.utility.looping as lp


def test1():
    pulse = context.init_pulselib(n_gates=1)

    s = pulse.mk_segment()

    amplitude = lp.linspace(100.0, 200.0, 5,
                            name='amplitude', unit='mV')
    s.P1.add_block(5, 15, amplitude)


    s.P1.add_ramp_ss(15, 18, 80, 0)
    s.reset_time()
    s.P1.add_sin(10, 40, 50, 350e6)

    s.wait(20)
    sequence = pulse.mk_sequence([s])
    sequence.n_rep = None

    for a in sequence.amplitude.values:
        sequence.amplitude(a)
        context.plot_awgs(sequence, analogue_out=True,
                          ylim=(-0.1, 0.250), xlim=(0, 80))


#%%
if __name__ == '__main__':
    ds1 = test1()
