from pulse_lib.base_pulse import pulselib
from example_init import return_pulse_lib
from loading_HVI_code_fast import load_HVI, set_and_compile_HVI, excute_HVI
# from loading_HVI_code import load_HVI, set_and_compile_HVI, excute_HVI

import numpy as np

pulse = return_pulse_lib()
pulse.cpp_uploader.resegment_memory()


PSB_pulse  = pulse.mk_segment()

# important points to pulse to
P1 = (0,0)
P2 = (500,500)
P3 = (300,500)
P4 = (200,500)

import pulse_lib.segments.utility.looping as lp
sweep_up_down = lp.linspace(-40,40,50, axis=0)
sweep_right_left = lp.linspace(-40,40,50, axis=1)

PSB_pulse.M2.add_marker(0,10)

PSB_pulse.P5.add_ramp_ss(0,100, 0,P1[0])
PSB_pulse.B4.add_ramp_ss(0,100, 0,P1[1])
PSB_pulse.reset_time()
PSB_pulse.P5.add_ramp_ss(0, 100, P1[0],P2[0])
PSB_pulse.B4.add_ramp_ss(0, 100, P1[1],P2[1])
PSB_pulse.reset_time()
PSB_pulse.P5.add_ramp_ss(0, 100, P2[0],P3[0])
PSB_pulse.B4.add_ramp_ss(0, 100, P2[1],P3[1])
PSB_pulse.reset_time()
PSB_pulse.P5.add_ramp_ss(0,100, P3[0],sweep_up_down + P4[0])
PSB_pulse.B4.add_ramp_ss(0,100, P3[1],sweep_right_left + P4[1])
PSB_pulse.reset_time()
PSB_pulse.M2.add_marker(0,10)
PSB_pulse.P5.add_block(0,10000, sweep_up_down + P4[0])
PSB_pulse.B4.add_block(0,10000, sweep_right_left + P4[1])
PSB_pulse.reset_time()
PSB_pulse.P5.add_ramp_ss(0,100, sweep_up_down + P4[0],P1[0])
PSB_pulse.B4.add_ramp_ss(0,100, sweep_right_left + P4[1],P1[1])
PSB_pulse.reset_time()
PSB_pulse.P5.wait(100)


# import matplotlib.pyplot as plt
# plt.figure()
# PSB_pulse.B4.plot_segment([0,0])
# PSB_pulse.P5.plot_segment([0,0])

# # PSB_pulse.M2.plot_segment([0,0])
# plt.xlabel("time (ns)")
# plt.ylabel("voltage (mV)")
# plt.legend()
# plt.show()


sequence = [PSB_pulse]

my_seq = pulse.mk_sequence(sequence)
my_seq.add_HVI(load_HVI, set_and_compile_HVI, excute_HVI)
my_seq.n_rep = 1000


import time
my_seq.neutralize = True
s = time.time()

for j in range(50):
	for i in range(50):
		my_seq.upload([i, j])
		my_seq.play([i, j])
		print(i,j)
e = time.time()

print("placback of 2500 waveforms (1000 repeat)",e-s, "seconds")
print("average number of uploads per second", 2500/(e-s))
pulse.uploader.wait_until_AWG_idle()
pulse.uploader.release_memory()
