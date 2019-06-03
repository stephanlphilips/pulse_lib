from pulse_lib.base_pulse import pulselib
from example_init import return_pulse_lib
# from loading_HVI_code_fast import load_HVI, set_and_compile_HVI, excute_HVI

import numpy as np

import matplotlib.pyplot as plt

def construct_ct(gate1, gate2, marker, t_step, vpp, n_pt):
	"""
	construct a the pulses needed for a charge stability diagram

	Args:
		gate1 (str) : gate in x direction to sweep
		gate2 (str) : gate in x direction to sweep
		marker (str) : marker which will be placed at each gate step
		t_step (float) : time step for a single point in the charge stability diagram (unit ns)
		vpp: (float) : peak to peak voltage 
		n_pt (int) : number of point along x/y (both are assumed to have the same number of points.)
	"""
	charge_st  = pulse.mk_segment()
	
	for  voltage in np.linspace(-vpp,vpp,n_pt):
	    getattr(charge_st, gate1).add_block(0, t_step, voltage)
	    getattr(charge_st, gate1).reset_time()
	getattr(charge_st, gate1).repeat(n_pt)


	for  voltage in np.linspace(-vpp,vpp,n_pt):
	    getattr(charge_st,gate2).add_block(0, t_step*n_pt, voltage)
	    getattr(charge_st,gate2).reset_time()

	getattr(charge_st,marker).add_marker(0,t_step/2)
	getattr(charge_st,marker).wait(t_step-t_step/2)
	getattr(charge_st,marker).repeat(n_pt)
	getattr(charge_st,marker).repeat(n_pt)

	getattr(charge_st,gate1).plot_segment(sample_rate = 1e6)
	getattr(charge_st,gate2).plot_segment(sample_rate = 1e6)
	getattr(charge_st,marker).plot_segment(sample_rate = 1e7)
	plt.show()

	return charge_st

if __name__ == '__main__':
	pulse, _ = return_pulse_lib()
	pulse.cpp_uploader.resegment_memory()
	sequence = [construct_ct("P4", "P5", "M2",1000 ,1000, 200)]

	my_seq = pulse.mk_sequence(sequence)

	my_seq.add_HVI(load_HVI, set_and_compile_HVI, excute_HVI)
	my_seq.n_rep = 100000
	print(my_seq.sample_rate)
	my_seq.sample_rate = 10e6
	print(my_seq.sample_rate)
	print(my_seq.prescaler)

	my_seq.upload([0])
	my_seq.play([0])

	pulse.uploader.wait_until_AWG_idle()
	pulse.uploader.release_memory()
