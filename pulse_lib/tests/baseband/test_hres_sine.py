
from pulse_lib.tests.configurations.test_configuration import context

#%%
import numpy as np
import matplotlib.pyplot as pt

from pulse_lib.qblox.pulsar_sequencers import PulsarConfig

def config_backend(pulse):
    if pulse._backend in ['Keysight', 'Keysight_QS']:
        context.station.AWG1.set_digital_filter_mode(3)
    if pulse._backend == 'Qblox':
        # increase resolution for fair check of algorithm
        PulsarConfig.NS_SUB_DIVISION = 20


def test0(t1, t2=10, hres=True):
    pulse = context.init_pulselib(n_gates=1)
    config_backend(pulse)

    s = pulse.mk_segment(hres=hres)
    P1 = s.P1

    P1.add_ramp_ss(0, 5, 0, 1000)
    P1.reset_time()
    P1.add_block(0, t1, 1000)
    P1.reset_time()
    P1.add_sin(0, t2, 1000, 500e6/t2, phase_offset=np.pi/2)
    P1.reset_time()
    P1.add_block(0, 4.0, -1000)
    P1.reset_time()
    P1.add_ramp_ss(0, 5, -1000, 0)
    P1.reset_time()

    s.wait(10)

    sequence = pulse.mk_sequence([s])
    sequence.n_rep = None

    context.plot_awgs(sequence, ylim=(-1.10, 1.10), xlim=(5, 30),
                      # analogue_out=True,
                      # analogue_shift=4.0-t1,
                      )


def test1(t1, t2=10, hres=True):
    pulse = context.init_pulselib(n_gates=2)
    config_backend(pulse)

    s = pulse.mk_segment(hres=hres)
    P1 = s.P1
    P2 = s.P2

    P1.add_ramp_ss(0, 5, 0, 100)
    P1.reset_time()
    P1.add_block(0, t1, 100)
    P1.reset_time()
    P1.add_sin(0, t2, 100, 500e6/t2, phase_offset=np.pi/2)
    P1.reset_time()
    P1.add_block(0, t1, -100)
    P1.reset_time()
    P1.add_ramp_ss(0, 5, -100, 0)
    P1.reset_time()

    P2.add_ramp_ss(0, 5, 0, 100)
    P2.reset_time()
    P2.add_block(0, round(t1), 100)
    P2.reset_time()
    P2.add_sin(0, round(t2), 100, 500e6/t2, phase_offset=np.pi/2)
    P2.reset_time()
    P2.add_block(0, round(2*t1)-round(t1), -100)
    P2.reset_time()
    P2.add_ramp_ss(0, 5, -100, 0)
    P2.reset_time()

    s.wait(10)

    sequence = pulse.mk_sequence([s])
    sequence.n_rep = None

    context.plot_awgs(sequence, ylim=(-0.110, 0.110), xlim=(5, 30),
                      # analogue_out=True,
                      )
    pt.title(f't1: {t1} t2: {t2}')


def test2(t1, hres=True):
    pulse = context.init_pulselib(n_gates=2)
    config_backend(pulse)

    s = pulse.mk_segment(hres=hres)
    P1 = s.P1
    # P2 = s.P2
    t2 = 10

    P1.add_ramp_ss(0, 5, 0, 100)
    P1.reset_time()
    P1.add_block(0, t1, 100)
    P1.reset_time()
    P1.add_block(0, t2, 100)
    P1.add_sin(0, t2, 100, 500e6/t2, phase_offset=np.pi)
    P1.reset_time()
    P1.add_block(0, t1, 100)
    P1.reset_time()
    P1.add_ramp_ss(0, 5, 100, 0)
    P1.reset_time()

    # P2.add_ramp_ss(0, 5, 0, 100)
    # P2.reset_time()
    # P2.add_block(0, round(t1), 100)
    # P2.reset_time()
    # P2.add_block(0, t2, 100)
    # P2.add_sin(0, round(t2), 100, 500e6/t2, phase_offset=np.pi)
    # P2.reset_time()
    # P2.add_block(0, round(2*t1)-round(t1), 100)
    # P2.reset_time()
    # P2.add_ramp_ss(0, 5, 100, 0)
    # P2.reset_time()

    s.wait(10)

    sequence = pulse.mk_sequence([s])
    sequence.n_rep = None

    context.plot_awgs(sequence, ylim=(-0.110, 0.110), xlim=(5, 30),
                      # analogue_out=True,
                      # analogue_shift=4.0-t1,
                      )
    pt.title(f't1: {t1} t2: {t2}')


def test3(t1, hres=True):
    pulse = context.init_pulselib(n_gates=1)
    config_backend(pulse)

    s = pulse.mk_segment(hres=hres)
    P1 = s.P1
    # P2 = s.P2
    t2 = 10

    P1.add_ramp_ss(0, 5, 0, 100)
    P1.reset_time()
    P1.add_block(0, t1, 100)
    P1.reset_time()
    P1.add_ramp_ss(0, 3, 100, 0)
    P1.reset_time()
    P1.add_sin(0, t2, 100, 500e6/t2, phase_offset=np.pi)
    P1.reset_time()
    P1.add_ramp_ss(0, 3, 0, 100)
    P1.reset_time()
    P1.add_block(0, t1, 100)
    P1.reset_time()
    P1.add_ramp_ss(0, 5, 100, 0)
    P1.reset_time()

    # P2.add_ramp_ss(0, 5, 0, 100)
    # P2.reset_time()
    # P2.add_block(0, round(t1), 100)
    # P2.reset_time()
    # P2.add_ramp_ss(0, 3, 100, 0)
    # P2.reset_time()
    # P2.add_sin(0, round(t2), 100, 500e6/t2, phase_offset=np.pi)
    # P2.reset_time()
    # P2.add_ramp_ss(0, 3, 0, 100)
    # P2.reset_time()
    # P2.add_block(0, round(2*t1)-round(t1), 100)
    # P2.reset_time()
    # P2.add_ramp_ss(0, 5, 100, 0)
    # P2.reset_time()

    s.wait(10)

    sequence = pulse.mk_sequence([s])
    sequence.n_rep = None

    context.plot_awgs(sequence, ylim=(-0.110, 0.110), xlim=(5, 30),
                      # analogue_out=True,
                      # analogue_shift=4.0-t1,
                      )
    pt.title(f't1: {t1} t2: {t2}')


def test4(t1, t2=10, hres=True):
    pulse = context.init_pulselib(n_gates=2)
    config_backend(pulse)

    s = pulse.mk_segment(hres=hres)
    P1 = s.P1
    P2 = s.P2

    P1.add_ramp_ss(0, 5, 0, 1000)
    P1.reset_time()
    P1.add_block(0, t1, 1000)
    P1.reset_time()
    P1.add_sin(0, t2, 1000, 500e6/t2, phase_offset=np.pi/2)
    P1.reset_time()
    P1.add_sin(0, t2/2, 1000, 500e6/t2, phase_offset=-np.pi/2)
    P1.reset_time()
    P1.add_sin(0, t2/2, 1000, 500e6/t2, phase_offset=0)
    P1.reset_time()
    P1.add_block(0, 4.0, 1000)
    P1.reset_time()
    P1.add_ramp_ss(0, 5, 1000, 0)
    P1.reset_time()

    P2.add_ramp_ss(0, 5, 0, 1000)
    P2.reset_time()
    P2.add_block(0, t1, 1000)
    P2.reset_time()
    P2.add_sin(0, 2*t2, 1000, 500e6/t2, phase_offset=np.pi/2)
    P2.reset_time()
    P2.add_block(0, 4.0, 1000)
    P2.reset_time()
    P2.add_ramp_ss(0, 5, 1000, 0)
    P2.reset_time()

    s.wait(10)

    sequence = pulse.mk_sequence([s])
    sequence.n_rep = None

    context.plot_awgs(sequence, ylim=(-1.10, 1.10), xlim=(5, 30),
                      # analogue_out=True,
                      # analogue_shift=4.0-t1,
                      )


#%%
if __name__ == '__main__':

    pt.figure()
    for t1 in [4.0, 4.2, 4.4, 4.6, 4.8]:
        test0(t1, t2=4.5, hres=True)

    for t1 in [4.0, 4.3, 4.49, 4.51, 4.8]:
        pt.figure()
        test1(t1, hres=False)
        pt.figure()
        test1(t1)

    pt.figure()
    for t1 in [4.0, 4.3, 4.49, 4.51, 4.8]:
        test2(t1, hres=False)

    pt.figure()
    for t1 in [4.0, 4.3, 4.49, 4.51, 4.8]:
        test3(t1, hres=False)
    pt.figure()
    for t1 in [4.0, 4.3, 4.49, 4.51, 4.8]:
        test3(t1)

    for t1 in [4.0, 4.2, 4.4, 4.6, 4.8]:
        pt.figure()
        test4(t1, t2=4.5, hres=True)

