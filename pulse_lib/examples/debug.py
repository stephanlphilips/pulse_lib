import numpy as np

from pulse_lib.base_pulse import pulselib
from example_init import return_pulse_lib
from charge_stability_diagram import construct_ct
# from loading_HVI_code_fast_single_shot import load_HVI, set_and_compile_HVI, excute_HVI
from loading_HVI_code_fast import load_HVI, set_and_compile_HVI, excute_HVI


import qcodes.instrument_drivers.Keysight.SD_common.SD_DIG_new_firmware as keysight_dig
import numpy as np
from pulse_lib.examples.loading_HVI_code_fast_single_shot import load_HVI, set_and_compile_HVI, excute_HVI
from pulse_lib.examples.loading_HVI_code_fast_single_shot_std_dig import load_HVI, set_and_compile_HVI, excute_HVI

import qcodes.instrument_drivers.Keysight.SD_common.SD_DIG_muti as keysight_dig

dig = keysight_dig.SD_DIG(name ="keysight_digitizer", chassis = 0, slot = 6, channels = 4, triggers=1)

dig.daq_flush(1)
dig.daq_flush(2)
dig.daq_flush(3)
dig.daq_flush(4)

dig.debug = False
def set_digitizer(digitizer1, npoints = 10, cycles = 1, Vmax = 2):
    """
    quick set of minumal settings to make it work.

    Args:
        digitizer1 (SD_DIG) : qcodes digitizer object
        npoints (int) : number of points to aquire in one cycle
        vmax (double) : maximum voltage of input (Vpeak)
    """
    # 1 point is 2 ns!
    digitizer1.data_mode(0)
    
    digitizer1.channels.ch1.set_channel_properties(Vmax)
    digitizer1.channels.ch2.set_channel_properties(Vmax)
    
    digitizer1.channels.ch1.set_daq_settings(cycles, npoints)
    digitizer1.channels.ch2.set_daq_settings(cycles, npoints)

# test param, aquire for 2us, repeat the experiment 1000 times
t_measure = 100
n_rep = 1000

set_digitizer(dig, npoints = 200, cycles = 50)
dig.meas_channel([1,2])
dig.daq_start(1)
dig.daq_start(2)
for i in range(50):
	dig.daq_trigger(1)
	dig.daq_trigger(2)

data = dig.measure()
print(data[0].shape, data[1].shape)