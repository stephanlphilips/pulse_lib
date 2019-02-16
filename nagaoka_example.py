from pulse_lib.base_pulse import pulselib

import numpy as np

p = pulselib()

class AWG(object):
	"""docstring for AWG"""
	def __init__(self, name):
		self.name = name
		self.chassis = 0
		self.slot = 0
		self.type = "DEMO"

AWG1 = AWG("AWG1")

# add to pulse_lib
p.add_awgs('AWG1',AWG1)

# define channels
awg_channels_to_physical_locations = dict({'P1':('AWG1', 1), 'P2':('AWG1', 2),
											'P3':('AWG1', 3), 'P4':('AWG1', 4),})
p.define_channels(awg_channels_to_physical_locations)

awg_virtual_gates = {
	'virtual_gates_names_virt' :
		['vP1','vP2','vP3','vP4'],
	'virtual_gates_names_real' :
		['P1','P2','P3','P4'],
	'virtual_gate_matrix' :
		np.eye(4)
}
p.add_virtual_gates(awg_virtual_gates)
p.add_channel_compenstation_limits({'P1': (-100,100),'P2': (-100,100),'P3': (-50,50),'P4': (-50,50)})
p.finish_init()

nagaoka_pulsing  = p.mk_segment()

base_level = 5 #mV

import pulse_lib.segments.looping as lp

ramp_amp = lp.linspace(50,80,20, axis=0)
ramp_speed = lp.linspace(5,30,20, axis=1)

nagaoka_pulsing.P1 += base_level
nagaoka_pulsing.P1.add_ramp(0,100,ramp_amp*.8)
nagaoka_pulsing.P1.reset_time()
nagaoka_pulsing.P1.add_block(0,ramp_speed,ramp_amp*.8)
nagaoka_pulsing.P1.add_ramp(0,ramp_speed,ramp_amp*.2)
nagaoka_pulsing.P1.reset_time()
nagaoka_pulsing.P1.add_block(0,100,ramp_amp)

nagaoka_pulsing.P1.append(nagaoka_pulsing.P1)
nagaoka_pulsing.P2 += nagaoka_pulsing.P1*0.23
nagaoka_pulsing.P3 += nagaoka_pulsing.P1*1.2
nagaoka_pulsing.P4 -= nagaoka_pulsing.P1*0.1

nagaoka_pulsing.P1.data[0,0].slice_time(45,125)
print(nagaoka_pulsing.P1.data[0,0].my_pulse_data)

import matplotlib.pyplot as plt
# plt.figure()
# nagaoka_pulsing.P1.plot_segment([0,0])
# nagaoka_pulsing.P2.plot_segment([0,0])
# nagaoka_pulsing.P3.plot_segment([0,0])
# nagaoka_pulsing.P4.plot_segment([0,0])

# plt.xlabel("time (ns)")
# plt.ylabel("voltage (mV)")
# plt.legend()
# plt.show()



import matplotlib.pyplot as plt
plt.figure()
nagaoka_pulsing.P1.plot_segment([0,0])
nagaoka_pulsing.P1.plot_segment([5,0])
nagaoka_pulsing.P1.plot_segment([5,5])
nagaoka_pulsing.P1.plot_segment([10,5])

plt.xlabel("time (ns)")
plt.ylabel("voltage (mV)")
plt.legend()
plt.show()

readout_level = 5
readout  = p.mk_segment()
readout.P1 += readout_level
readout.P1.wait(1e6)

# sequence using default settings
sequence = [nagaoka_pulsing, readout]

my_seq = p.mk_sequence(sequence)

# my_seq.upload([i])
# for i in range(len(my_seq)):
# 	my_seq.play([i])
# 	if i < len(my_seq)-1:
# 		my_seq.upload([i+1])
# 	# commands to get data
# 	# this should be put in an experiment class
# 	# and handled by qcodes
