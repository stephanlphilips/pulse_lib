# from pulse_lib.base_pulse import pulselib
from example_init import return_pulse_lib
# # from loading_HVI_code_fast_single_shot import load_HVI, set_and_compile_HVI, excute_HVI
# # from loading_HVI_code_fast import load_HVI, set_and_compile_HVI, excute_HVI

# # from loading_HVI_code import load_HVI, set_and_compile_HVI, excute_HVI
# import qcodes.instrument_drivers.Keysight.SD_common.SD_DIG_new_firmware as keysight_dig

# import numpy as np

# dig = keysight_dig.SD_DIG(name ="keysight_digitizer", chassis = 0, slot = 6, channels = 4, triggers=1)

# pulse, virtual_gate = return_pulse_lib()
# pulse.cpp_uploader.resegment_memory()


import time 


import pulse_lib.segments.utility.looping as lp
from pulse_lib.segments.segment_container import segment_container
import matplotlib.pyplot as plt

pulse, virtual_gate = return_pulse_lib()

PSB_pulse  = pulse.mk_segment()

# important points to pulse to
P0 = (-100,-100)
P1 = (-100,100)
P2 = (-25,25)
P3 = (-5,-5)
P4 = (25, 25)

import pulse_lib.segments.utility.looping as lp
P4_Point_3 = lp.linspace(P2[0],P3[0],5, axis=0, unit = "mV", name = "vP4")
P5_Point_3 = lp.linspace(P2[1],P3[1],5, axis=1, unit = "mV", name = "vP5")


PSB_pulse.vP4.add_block(0,10000, P0[0])
PSB_pulse.vP5.add_block(0,10000, P0[1])
PSB_pulse.reset_time()
PSB_pulse.vP4.add_block(0,10000, P1[0])
PSB_pulse.vP5.add_block(0,10000, P1[1])
PSB_pulse.reset_time()
PSB_pulse.vP4.add_ramp_ss(0,500, P1[0], P2[0])
PSB_pulse.vP5.add_ramp_ss(0,500, P1[1], P2[1])
PSB_pulse.reset_time()
PSB_pulse.vP4.add_ramp_ss(0,500, P2[0], P4_Point_3)
PSB_pulse.vP5.add_ramp_ss(0,500, P2[1], P5_Point_3)
PSB_pulse.reset_time()
PSB_pulse.vP4.add_block(0,50000, P4_Point_3)
PSB_pulse.vP5.add_block(0,50000, P5_Point_3)
PSB_pulse.reset_time()
PSB_pulse.vP4.add_block(0,1000, P0[0])
PSB_pulse.vP5.add_block(0,1000, P0[1])

# PSB_pulse.P4.plot_segment([1], render_full=True)
# PSB_pulse.P5.plot_segment([1,0], render_full=True)
# plt.show()
import copy
my_seq = [PSB_pulse, copy.copy(PSB_pulse)] #,copy.copy(PSB_pulse),copy.copy(PSB_pulse),copy.copy(PSB_pulse),copy.copy(PSB_pulse)]
seq = pulse.mk_sequence(my_seq)

