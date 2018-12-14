from pulse_lib.base_pulse import pulselib
import numpy as np


p = pulselib()

# add to pulse_lib
p.add_awgs('AWG1',None)
p.add_awgs('AWG2',None)
p.add_awgs('AWG3',None)
p.add_awgs('AWG4',None)

# define channels
awg_channels_to_physical_locations = dict({'B0':('AWG1', 1), 'P1':('AWG1', 2),
	'B1':('AWG1', 3), 'P2':('AWG1', 4),
	'B2':('AWG2', 1), 'P3':('AWG2', 2),
	'B3':('AWG2', 3), 'P4':('AWG2', 4),
	'B4':('AWG3', 1), 'P5':('AWG3', 2),
	'B5':('AWG3', 3), 'G1':('AWG3', 4),
	'I_MW':('AWG4', 1), 'Q_MW':('AWG4', 2),	
	'M1':('AWG4', 3), 'M2':('AWG4', 4)})
	
p.define_channels(awg_channels_to_physical_locations)

# format : dict of channel name with delay in ns (can be posive/negative)
p.add_channel_delay({'I_MW':50, 'Q_MW':50, 'M1':20, 'M2':-25, })

awg_virtual_gates = {'virtual_gates_names_virt' :
	['vP1','vP2','vP3','vP4','vP5','vB0','vB1','vB2','vB3','vB4','vB5'],
			'virtual_gates_names_real' :
	['P1','P2','P3','P4','P5','B0','B1','B2','B3','B4','B5'],
	 'virtual_gate_matrix' : np.eye(11)
}
p.add_virtual_gates(awg_virtual_gates)

awg_IQ_channels = {'vIQ_channels' : ['qubit_1','qubit_2'],
			'rIQ_channels' : [['I_MW','Q_MW'],['I_MW','Q_MW']],
			'LO_freq' :[1.01e9, 1e9]
			# do not put the brackets for the MW source
			# e.g. MW_source.frequency
			}
	
p.add_IQ_virt_channels(awg_IQ_channels)
p.finish_init()

seg  = p.mk_segment('INIT')


# B0 is the barrier 0 channel
# adds a linear ramp from 10 to 20 ns with amplitude of 5 to 10.
# seg.B0.add_pulse([[10.,0.],[10.,5.],[20.,10.],[20.,0.]])
# add a block pulse of 2V from 40 to 70 ns, to whaterver waveform is already there
seg.B0.add_block(0,1000,2)

import pulse_lib.segments.looping as lp


times = lp.linspace(5,20e3, 20, axis=0, name="time", unit="ns")
# test = lp.loop_obj()

# data = np.linspace(0,5,1000)
# data = np.resize(data, (10,100))

# test.data = data
# test.names = ["a","b"]
# test.units = ["a","b"]
# test.axis = [-1,-1]


# just waits (e.g. you want to ake a segment 50 ns longer)
# times = lp.linspace(50, 80)
seg.B0.wait(times)
# resets time back to zero in segment. Al the commannds we run before will be put at a negative time.
seg.reset_time()
# this pulse will be placed directly after the wait()
# seg.B0.add_pulse([[10.,0.],[10.,5.],[20.,10.],[20.,2.]])
# seg.B0.add_block(0,1000,-2)

print(seg.B0.data.shape)
print(seg.B0.total_time)
print(seg.B1.total_time)
# print(seg.B0.v_min([0]))
# print(seg.B0.v_max([0]))
# print(seg.B0.integrate([0]))
# print(seg.B1.data.shape)
# seg.B0.plot_segment( render_full=False)

# print(seg.total_time)

