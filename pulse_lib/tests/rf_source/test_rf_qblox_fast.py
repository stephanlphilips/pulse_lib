
from pulse_lib.tests.configurations.test_configuration import context
from pulse_lib.fast_scan.qblox_fast_scans import fast_scan1D_param, fast_scan2D_param

#%%
def test():
    pulse = context.init_pulselib(n_gates=2, n_sensors=2, rf_sources=True)

    param1 = fast_scan1D_param(
            pulse, 'P1', 100.0, 10, 5000,
            channels=['SD2'])

    context.plot_segments(param1.my_seq.sequence)
    context.plot_awgs(param1.my_seq)

    param2 = fast_scan2D_param(
            pulse, 'P1', 100.0, 10, 'P2', 80.0, 10, 2000,
            channels=['SD2'])

    context.plot_segments(param2.my_seq.sequence)
    context.plot_awgs(param2.my_seq)

    return None

if __name__ == '__main__':
    ds = test()
