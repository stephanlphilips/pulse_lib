from pulse_lib.base_pulse import pulselib
import numpy as np

def return_pulse_lib():
	pulse = pulselib()


	# Let's just use a non-pysical AWG
	pulse.add_awgs('AWG1',None)
	pulse.add_awgs('AWG2',None)


	# define real channels
	awg_channels_to_physical_locations = dict({'B0':('AWG1', 1), 'P1':('AWG1', 2),
	        'B1':('AWG1', 3), 'P2':('AWG1', 4),'B2':('AWG2', 1),
	        'MW_gate_I':('AWG2', 2), 'MW_gate_Q':('AWG2', 3),
	        'MW_marker':('AWG2', 4)})

	pulse.define_channels(awg_channels_to_physical_locations)

	# define virtual channels
	awg_virtual_gates = {
	        'virtual_gates_names_virt' :
	                ['vB0', 'vB1', 'vB2', 'vP1', 'vP2'],
	        'virtual_gates_names_real' :
	                ['B0', 'B1', 'B2', 'P1', 'P2'],
	        'virtual_gate_matrix' : np.eye(5)
	}
	pulse.add_virtual_gates(awg_virtual_gates)

	# define IQ channels
	awg_IQ_channels = {
	                'vIQ_channels' : ['qubit_1','qubit_2'],
	                'rIQ_channels' : [['MW_gate_I','MW_gate_Q'],['MW_gate_I','MW_gate_Q']],
	                'LO_freq' :[10e9, 10e9]
	                }

	pulse.add_IQ_virt_channels(awg_IQ_channels)

	# define delays
	pulse.add_channel_delay({
	        'B0': 20,
	        'P1': 20,
	        'B1': 20,
	        'P2': 20,
	        'B2': 20,
	        'MW_gate_I': 70,
	        'MW_gate_Q': 70,
	        'MW_marker': 5
	})

	# add compensation limits
	pulse.add_channel_compenstation_limits({
		'B0': (-500,500),'B1': (-500,500),'B2': (-500,500),
		'P1': (-500,500),'P2': (-500,500),
		})
	# finish initialisation (! important if using keysight uploader)
	pulse.finish_init()

	return pulse

if __name__ == '__main__':
	pulse = return_pulse_lib()