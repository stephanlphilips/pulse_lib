import numpy as np
import matplotlib.pyplot as pt

from pulse_lib.tests.hw_schedule_mock import HardwareScheduleMock

from configuration.small import init_hardware, init_pulselib

from utils.plot import plot_awgs


def create_2D_scan(pulse_lib, gate1, sweep1_mv, gate2, sweep2_mv, n_steps, t_measure,
                   bias_T_corr=False):


    seg = pulse_lib.mk_segment()
    p1 = seg[gate1]
    p2 = seg[gate2]

    t_sweep = n_steps*t_measure*1000

    for i in range(n_steps):
        p1.add_ramp_ss(0, t_sweep, -sweep1_mv, sweep1_mv)
        p1.reset_time()

    v_steps = np.linspace(-sweep2_mv, sweep2_mv, n_steps)
    if bias_T_corr:
        v2 = np.zeros((n_steps))
        mid = (n_steps+1)//2
        v2[::2] = v_steps[:mid]
        v2[1::2] = v_steps[mid:][::-1]
    else:
        v2 = v_steps

    for v_step in v2:
        p2.add_block(0, t_sweep, v_step)
        p2.reset_time()

    # generate the sequence from segments
    my_seq = pulse_lib.mk_sequence([seg])
    my_seq.set_hw_schedule(HardwareScheduleMock())
    my_seq.n_rep = 1
    # 2 points per voltage step
    my_seq.sample_rate = 2 / (t_measure * 1e-6)

    return my_seq


# create "AWG1"
awgs = init_hardware()

# create channels P1, P2
p = init_pulselib(awgs, virtual_gates=True)


my_seq = create_2D_scan(
        p,
        'P1', 100,
        'P2', 100,
        100, 50.0,
        bias_T_corr=True
        )

my_seq.upload()
my_seq.play()

# RC = 100 ms:  cut-off frequency, fc = 1/(2 pi RC) = 1.6 Hz
bias_T_rc_time = 0.1
#bias_T_rc_time = None

plot_awgs(awgs, bias_T_rc_time=bias_T_rc_time)
pt.title('AWG upload (with DC compensation)')
pt.grid(True)


