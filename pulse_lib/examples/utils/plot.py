
import matplotlib.pyplot as pt

def plot_awgs(awgs, bias_T_rc_time=None):
    pt.figure()

    for awg in awgs:
        awg.plot(bias_T_rc_time=bias_T_rc_time)

    pt.legend()
    pt.ylabel('amplitude [V]')
    pt.xlabel('time [ns]')
    pt.title('AWG upload')
