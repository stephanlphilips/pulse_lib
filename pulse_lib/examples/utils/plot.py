
import matplotlib.pyplot as pt

def plot_awgs(awgs, bias_T_rc_time=None):
    do_plot = False
    for awg in awgs:
        if hasattr(awg, 'plot'):
            do_plot = True

    if not do_plot:
        return

    pt.figure()

    for awg in awgs:
        if hasattr(awg, 'plot'):
            awg.plot(bias_T_rc_time=bias_T_rc_time)

    pt.legend()
    pt.ylabel('amplitude [V]')
    pt.xlabel('time [ns]')
    pt.title('AWG upload')
