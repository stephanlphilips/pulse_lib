from pulse_lib.base_pulse import pulselib
from example_init import return_pulse_lib
from loading_HVI_code_fast import load_HVI, set_and_compile_HVI, excute_HVI
import numpy as np

pulse = return_pulse_lib()

import time

# t1 = time.time()
pulse.cpp_uploader.resegment_memory()
# t2 = time.time(z)
# print(t2-t1)

nagaoka_pulsing  = pulse.mk_segment()

# base_level = 5 #mV

import pulse_lib.segments.utility.looping as lp

ramp_amp = lp.linspace(50,200,50, axis=0)
ramp_speed = lp.linspace(5,100,50, axis=1)

# nagaoka_pulsing.B4 += base_level
nagaoka_pulsing.B4.add_block(0,50,1000)
nagaoka_pulsing.B4.reset_time()
nagaoka_pulsing.B4.add_block(0,50,900)
nagaoka_pulsing.B4.reset_time()
nagaoka_pulsing.B4.add_block(0,100,100)
nagaoka_pulsing.B4.reset_time()
nagaoka_pulsing.B4.add_ramp(0,100,ramp_amp*.8)
nagaoka_pulsing.B4.reset_time()
nagaoka_pulsing.B4.add_block(0,ramp_speed,ramp_amp*.8)
nagaoka_pulsing.B4.add_ramp(0,ramp_speed,ramp_amp*.2)
# nagaoka_pulsing.B4.add_block(0,2e3,0)
nagaoka_pulsing.B4.reset_time()
nagaoka_pulsing.B4.add_block(0,100,500)
# nagaoka_pulsing.B4.reset_time()
# nagaoka_pulsing.B4.add_block(0,100,0)
# nagaoka_pulsing.B4.reset_time()
# nagaoka_pulsing.P2 += nagaoka_pulsing.P1
# nagaoka_pulsing.B4 += nagaoka_pulsing.P1
# nagaoka_pulsing.B1 -= nagaoka_pulsing.P1

nagaoka_pulsing.B5 = nagaoka_pulsing.B4
 
import matplotlib.pyplot as plt
# plt.figure()
# nagaoka_pulsing.B4.plot_segment([0,0])
# nagaoka_pulsing.P2.plot_segment([0,0])
# nagaoka_pulsing.P3.plot_segment([0,0])
# nagaoka_pulsing.P4.plot_segment([0,0])
# plt.xlabel("time (ns)")
# plt.ylabel("voltage (mV)")
# plt.legend()
# plt.show()



# import matplotlib.pyplot as plt
# plt.figure()
# nagaoka_pulsing.B4.plot_segment([0,0])
# nagaoka_pulsing.B4.plot_segment([1,0])
# nagaoka_pulsing.B4.plot_segment([2,0])
# nagaoka_pulsing.B4.plot_segment([3,0])

# plt.xlabel("time (ns)")
# plt.ylabel("voltage (mV)")
# plt.legend()
# plt.show()

readout_level = -200
readout  = pulse.mk_segment()
# readout.P1 += readout_level
# readout.P1.wait(2e3)
# readout.B4.add_block(0,2e3,0)
readout.B4.add_block(0,8e3,100)
readout.B4.reset_time()
# readout.B4.add_block(0,8e3,80)
# readout.B4.reset_time()
# readout.B4.add_block(0,8e3,60)
# readout.B4.reset_time()
# readout.B4.add_block(0,8e3,40)

# sequence using default settings
sequence = [nagaoka_pulsing, readout]

my_seq = pulse.mk_sequence(sequence)
my_seq.add_HVI(load_HVI, set_and_compile_HVI, excute_HVI)
my_seq.n_rep = 1000

# my_seq.upload([0])
# time.sleep(0.03)
# my_seq.play([0])

import time
my_seq.neutralize = False
s = time.time()

for j in range(50):
	for i in range(50):
		my_seq.upload([i, j])
		my_seq.play([i, j])

pulse.uploader.wait_until_AWG_idle()
pulse.uploader.release_memory()

# for j in range(50):
# 	for i in range(50):
# 		my_seq.upload([i, j])
# 		my_seq.play([i, j])

# pulse.uploader.wait_until_AWG_idle()
# pulse.uploader.release_memory()
# for j in range(50):
# 	for i in range(50):
# 		my_seq.upload([i, j])
# 		my_seq.play([i, j])

# pulse.uploader.wait_until_AWG_idle()
# pulse.uploader.release_memory()

# for j in range(50):
# 	for i in range(50):
# 		my_seq.upload([i, j])
# 		my_seq.play([i, j])

# pulse.uploader.wait_until_AWG_idle()
# pulse.uploader.release_memory()

# for j in range(50):
# 	for i in range(50):
# 		my_seq.upload([i, j])
# 		my_seq.play([i, j])

# pulse.uploader.wait_until_AWG_idle()
# pulse.uploader.release_memory()

# for j in range(50):
# 	for i in range(50):
# 		my_seq.upload([i, j])
# 		my_seq.play([i, j])

# pulse.uploader.wait_until_AWG_idle()
# pulse.uploader.release_memory()

# for j in range(50):
# 	for i in range(50):
# 		my_seq.upload([i, j])
# 		my_seq.play([i, j])

# pulse.uploader.wait_until_AWG_idle()
# pulse.uploader.release_memory()

# for j in range(50):
# 	for i in range(50):
# 		my_seq.upload([i, j])
# 		my_seq.play([i, j])

# pulse.uploader.wait_until_AWG_idle()
# pulse.uploader.release_memory()

# for j in range(50):
# 	for i in range(50):
# 		my_seq.upload([i, j])
# 		my_seq.play([i, j])

# pulse.uploader.wait_until_AWG_idle()
# pulse.uploader.release_memory()

# for j in range(50):
# 	for i in range(50):
# 		my_seq.upload([i, j])
# 		my_seq.play([i, j])

# pulse.uploader.wait_until_AWG_idle()
# pulse.uploader.release_memory()

# for j in range(50):
# 	for i in range(50):
# 		my_seq.upload([i, j])
# 		my_seq.play([i, j])

# pulse.uploader.wait_until_AWG_idle()
# pulse.uploader.release_memory()