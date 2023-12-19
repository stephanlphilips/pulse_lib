
from pulse_lib.tests.configurations.test_configuration import context

#%%
import numpy as np
import matplotlib.pyplot as pt


def test1(t1, t2=10, hres=True):
    pulse = context.init_pulselib(n_gates=2)
    context.station.AWG1.set_digital_filter_mode(3)

    s = pulse.mk_segment(hres=hres)
    P1 = s.P1
    P2 = s.P2

    P1.add_ramp_ss(0, 5, 0, 100)
    P1.reset_time()
    P1.add_block(0, t1, 100)
    P1.reset_time()
    P1.add_sin(0, t2, 100, 50e6, phase_offset=np.pi/2)
    P1.reset_time()
    P1.add_block(0, t1, -100)
    P1.reset_time()
    P1.add_ramp_ss(0, 5, -100, 0)
    P1.reset_time()

    P2.add_ramp_ss(0, 5, 0, 100)
    P2.reset_time()
    P2.add_block(0, round(t1), 100)
    P2.reset_time()
    P2.add_sin(0, round(t2), 100, 50e6, phase_offset=np.pi/2)
    P2.reset_time()
    P2.add_block(0, round(2*t1)-round(t1), -100)
    P2.reset_time()
    P2.add_ramp_ss(0, 5, -100, 0)
    P2.reset_time()

    s.wait(10)

    sequence = pulse.mk_sequence([s])
    sequence.n_rep = None

    context.plot_awgs(sequence, ylim=(-0.110, 0.110), xlim=(5, 30), analogue_out=True)
    pt.title(f't1: {t1} t2: {t2}')


def test2(t1, hres=True):
    pulse = context.init_pulselib(n_gates=2)
    context.station.AWG1.set_digital_filter_mode(3)

    s = pulse.mk_segment(hres=hres)
    P1 = s.P1
    P2 = s.P2
    t2 = 10

    P1.add_ramp_ss(0, 5, 0, 100)
    P1.reset_time()
    P1.add_block(0, t1, 100)
    P1.reset_time()
    P1.add_block(0, t2, 100)
    P1.add_sin(0, t2, 100, 50e6, phase_offset=np.pi)
    P1.reset_time()
    P1.add_block(0, t1, 100)
    P1.reset_time()
    P1.add_ramp_ss(0, 5, 100, 0)
    P1.reset_time()

    P2.add_ramp_ss(0, 5, 0, 100)
    P2.reset_time()
    P2.add_block(0, round(t1), 100)
    P2.reset_time()
    P2.add_block(0, t2, 100)
    P2.add_sin(0, round(t2), 100, 50e6, phase_offset=np.pi)
    P2.reset_time()
    P2.add_block(0, round(2*t1)-round(t1), 100)
    P2.reset_time()
    P2.add_ramp_ss(0, 5, 100, 0)
    P2.reset_time()

    s.wait(10)

    sequence = pulse.mk_sequence([s])
    sequence.n_rep = None

    context.plot_awgs(sequence, ylim=(-0.110, 0.110), xlim=(5, 30), analogue_out=True)
    pt.title(f't1: {t1} t2: {t2}')


def test3(t1, hres=True):
    pulse = context.init_pulselib(n_gates=2)
    context.station.AWG1.set_digital_filter_mode(3)

    s = pulse.mk_segment(hres=hres)
    P1 = s.P1
    P2 = s.P2
    t2 = 10

    P1.add_ramp_ss(0, 5, 0, 100)
    P1.reset_time()
    P1.add_block(0, t1, 100)
    P1.reset_time()
    P1.add_ramp_ss(0, 3, 100, 0)
    P1.reset_time()
    P1.add_sin(0, t2, 100, 50e6, phase_offset=np.pi)
    P1.reset_time()
    P1.add_ramp_ss(0, 3, 0, 100)
    P1.reset_time()
    P1.add_block(0, t1, 100)
    P1.reset_time()
    P1.add_ramp_ss(0, 5, 100, 0)
    P1.reset_time()

    P2.add_ramp_ss(0, 5, 0, 100)
    P2.reset_time()
    P2.add_block(0, round(t1), 100)
    P2.reset_time()
    P2.add_ramp_ss(0, 3, 100, 0)
    P2.reset_time()
    P2.add_sin(0, round(t2), 100, 50e6, phase_offset=np.pi)
    P2.reset_time()
    P2.add_ramp_ss(0, 3, 0, 100)
    P2.reset_time()
    P2.add_block(0, round(2*t1)-round(t1), 100)
    P2.reset_time()
    P2.add_ramp_ss(0, 5, 100, 0)
    P2.reset_time()

    s.wait(10)

    sequence = pulse.mk_sequence([s])
    sequence.n_rep = None

    context.plot_awgs(sequence, ylim=(-0.110, 0.110), xlim=(5, 30))#, analogue_out=True)
    pt.title(f't1: {t1} t2: {t2}')


#%%
if __name__ == '__main__':
    for t1 in [4.3, 4.49, 4.51, 4.8]:
        test1(t1, hres=False)
        test1(t1)

    for t1 in [4.3, 4.49, 4.51, 4.8]:
        test2(t1, hres=False)
        test2(t1)

    for t1 in [4.0, 4.3, 4.49, 4.51, 4.8]:
        test3(t1, hres=False)
        test3(t1)
