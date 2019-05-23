# from pulse_lib.base_pulse import pulselib
# from example_init import return_pulse_lib
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
PSB_pulse  = segment_container(["vP4", "vP5"])

# important points to pulse toprep
P0 = (-150,-100)
P1 = (-150,100)
P2 = (-12.5,12.5)
P3 = (75,-75)
P4 = (5, 30)

P4_Point_3 = lp.linspace(P2[0],P3[0],100, axis=0, unit = "mV", name = "setpoints AWG")
P5_Point_3 = lp.linspace(P2[1],P3[1],100, axis=1, unit = "mV", name = "axis 2 sweep")

import time
t1 = time.time()
PSB_pulse.vP4.add_block(0,10000, P0[0])
PSB_pulse.vP5.add_block(0,10000, P0[1])
PSB_pulse.reset_time()
PSB_pulse.vP4.add_block(0,10000, P1[0])
PSB_pulse.vP5.add_block(0,10000, P1[1])
PSB_pulse.reset_time()
PSB_pulse.vP4.add_ramp_ss(0,100, P1[0], P2[0])
PSB_pulse.vP5.add_ramp_ss(0,100, P1[1], P2[1])
PSB_pulse.reset_time()
PSB_pulse.vP4.add_ramp_ss(0,100, P2[0], P4_Point_3)
PSB_pulse.vP5.add_ramp_ss(0,100, P2[1], P5_Point_3)
PSB_pulse.reset_time()
PSB_pulse.vP4.add_block(0,1000, P2[0])
PSB_pulse.vP5.add_block(0,1000, P2[1])
PSB_pulse.reset_time()
t2 = time.time()
print(t2-t1)
# 3.7 ms generation time --> OK!

# PSB_pulse.vP4.plot_segment([0])
# PSB_pulse.vP4.plot_segment([80])
# plt.show()


# PSB_pulse.P5.add_ramp_ss(0, 100, P1[0],P2[0])
# PSB_pulse.B4.add_ramp_ss(0, 100, P1[1],P2[1])
# PSB_pulse.reset_time()
# PSB_pulse.P5.add_ramp_ss(0, 100, P2[0],P3[0])
# PSB_pulse.B4.add_ramp_ss(0, 100, P2[1],P3[1])
# PSB_pulse.reset_time()
# PSB_pulse.P5.add_ramp_ss(0,100, P3[0],sweep_up_down + P4[0])
# PSB_pulse.B4.add_ramp_ss(0,100, P3[1],sweep_right_left + P4[1])
# PSB_pulse.reset_time()
# PSB_pulse.M2.add_marker(0,10)
# PSB_pulse.P5.add_block(0,10000, sweep_up_down + P4[0])
# PSB_pulse.B4.add_block(0,10000, sweep_right_left + P4[1])
# PSB_pulse.reset_time()
# PSB_pulse.P5.add_ramp_ss(0,100, sweep_up_down + P4[0],P1[0])
# PSB_pulse.B4.add_ramp_ss(0,100, sweep_right_left + P4[1],P1[1])
# PSB_pulse.reset_time()
# PSB_pulse.P5.wait(100)


# import matplotlib.pyplot as plt
# # plt.figure()
# # PSB_pulse.B4.plot_segment([0,0])
# # PSB_pulse.P5.plot_segment([0,0])

# PSB_pulse.A2.plot_segment()
# PSB_pulse.A4.plot_segment()
# PSB_pulse.A6.plot_segment()

# plt.xlabel("time (ns)")
# plt.ylabel("voltage (mV)")
# plt.legend()
# plt.show()

# def construct_ct(gate1, gate2, marker, t_step, vpp, n_pt):
# 	"""
# 	construct a the pulses needed for a charge stability diagram

# 	Args:
# 		gate1 (str) : gate in x direction to sweep
# 		gate2 (str) : gate in x direction to sweep
# 		marker (str) : marker which will be placed at each gate step
# 		t_step (float) : time step for a single point in the charge stability diagram (unit ns)
# 		vpp: (float) : peak to peak voltage 
# 		n_pt (int) : number of point along x/y (both are assumed to have the same number of points.)
# 	"""
# 	charge_st  = pulse.mk_segment()

# 	for  voltage in np.linspace(-vpp,vpp,n_pt):
# 	    getattr(charge_st, gate1).add_block(0, t_step, voltage)
# 	    getattr(charge_st, gate1).reset_time()
# 	getattr(charge_st, gate1).repeat(n_pt)


# 	for  voltage in np.linspace(-vpp,vpp,n_pt):
# 	    getattr(charge_st,gate2).add_block(0, t_step*n_pt, voltage)
# 	    getattr(charge_st,gate2).reset_time()

# 	# getattr(charge_st,marker).add_marker(0,t_step/2)
# 	# getattr(charge_st,marker).wait(t_step-t_step/2)
# 	# getattr(charge_st,marker).repeat(n_pt)
# 	# getattr(charge_st,marker).repeat(n_pt)

# 	return charge_st

# sequence = [PSB_pulse]
# # sequence = [PSB_pulse, construct_ct("A6", "A2", "A4",1000 ,1000, 30)]
# my_seq = pulse.mk_sequence(sequence)
# my_seq.add_HVI(load_HVI, set_and_compile_HVI, excute_HVI, digitizer = dig, dig_wait = 100)
# my_seq.n_rep = 1000000000


# import time
# my_seq.neutralize = False

# my_seq.upload([0])
# my_seq.play([0])

# # # for j in range(50):
# # # 	for i in range(50):
# # # 		my_seq.upload([i, j])
# # # 		my_seq.play([i, j])
# # # 		print(i,j)
# # # e = time.time()

# # # print("placback of 2500 waveforms (1000 repeat)",e-s, "seconds")
# # # print("average number of uploads per second", 2500/(e-s))
# pulse.uploader.wait_until_AWG_idle()
# pulse.uploader.release_memory()
