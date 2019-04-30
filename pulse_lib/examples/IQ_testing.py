from pulse_lib.base_pulse import pulselib
from example_init import return_pulse_lib
from loading_HVI_code_fast import load_HVI, set_and_compile_HVI, excute_HVI
import numpy as np

pulse = return_pulse_lib()



pulse.cpp_uploader.resegment_memory()


my_new_segment  = pulse.mk_segment()
print(my_new_segment.channels)
# my_new_segment.B0.add_block(0,100,200)
# my_new_segment.reset_time()
# my_new_segment.B0.add_block(0,2000,-600)
# my_new_segment.reset_time()
# my_new_segment.B0.add_block(0,2000,600)
# my_new_segment.reset_time()
my_new_segment.B0.add_block(0,10,500)
my_new_segment.B0.reset_time()
my_new_segment.B0.add_block(0,10,500)
my_new_segment.B0.reset_time()
my_new_segment.B0.add_block(0,100,100)
my_new_segment.B0.reset_time()
my_new_segment.B0.add_ramp(0,100,100*.8)
my_new_segment.B0.reset_time()
my_new_segment.B0.add_block(0,50,100*.8)
my_new_segment.B0.add_ramp(0,50,100*.2)

# my_new_segment.MW_marker.add_block(0,600,1000)
# my_new_segment.qubit_1.add_sin(100,500,1.01e9,500)
# my_new_segment.qubit_2.add_sin(100,500,1.01e9,1000)




sequence = [my_new_segment]

my_seq = pulse.mk_sequence(sequence)
my_seq.neutralize = False
my_seq.add_HVI(load_HVI, set_and_compile_HVI, excute_HVI)
my_seq.n_rep = 0


# for j in range(50):
# 	for i in range(50):
# 		my_seq.upload([i, j])
# 		my_seq.play([i, j])
my_seq.upload([0])
my_seq.play([0])
pulse.uploader.wait_until_AWG_idle()
pulse.uploader.release_memory()