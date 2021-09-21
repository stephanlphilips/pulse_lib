
from pulse_lib.base_pulse import pulselib
from pulse_lib.virtual_channel_constructors import IQ_channel_constructor, virtual_gates_constructor

# import qcodes.instrument_drivers.Keysight.SD_common.SD_AWG as keysight_awg
# import qcodes.instrument_drivers.Keysight.SD_common.SD_DIG as keysight_dig
import numpy as np

def return_pulse_lib():
	pulse = pulselib()

	# add to pulse_lib
#	pulse.add_awgs('AWG1',None)
#	pulse.add_awgs('AWG2',None)
#	pulse.add_awgs('AWG3',None)
#	pulse.add_awgs('AWG4',None)

	# define channels
	pulse.define_channel('B3','AWG1', 1)
	pulse.define_channel('B4','AWG1', 2)
	pulse.define_channel('P5','AWG1', 3)
	pulse.define_channel('P4','AWG1', 4)
	pulse.define_channel('P1','AWG2', 1)
	pulse.define_channel('B1','AWG2', 2)
	pulse.define_channel('P2','AWG2', 3)
	pulse.define_channel('B0','AWG2', 4)
	pulse.define_channel('B5','AWG3', 1)
	pulse.define_channel('A2','AWG3', 2)
	pulse.define_channel('A3','AWG3', 3)
	pulse.define_channel('A4','AWG3', 4)
	pulse.define_channel('A5','AWG4',1)
	pulse.define_channel('M1','AWG4',2)
	pulse.define_channel('A6','AWG4', 3)
	pulse.define_marker('M2','AWG4', 4)


	# format : channel name with delay in ns (can be posive/negative)
	# pulse.add_channel_delay('I_MW',50)
	# pulse.add_channel_delay('Q_MW',50)
	# pulse.add_channel_delay('M1',20)
	# pulse.add_channel_delay('M2',-25)

	# add limits on voltages for DC channel compenstation (if no limit is specified, no compensation is performed).
	pulse.add_channel_compenstation_limit('B0', (-1000, 500))
	pulse.add_channel_compenstation_limit('B1', (-1000, 500))
	pulse.add_channel_compenstation_limit('B3', (-1000, 500))
	pulse.add_channel_compenstation_limit('B4', (-1000, 500))
	pulse.add_channel_compenstation_limit('P1', (-1000, 500))
	pulse.add_channel_compenstation_limit('P2', (-1000, 500))
	pulse.add_channel_compenstation_limit('P4', (-1000, 500))
	pulse.add_channel_compenstation_limit('P5', (-1000, 500))
	pulse.add_channel_compenstation_limit('B5', (-1000, 500))
	# set a virtual gate matrix (note that you are not limited to one matrix if you would which so)
	virtual_gate_set_1 = virtual_gates_constructor(pulse)
	virtual_gate_set_1.add_real_gates('P4','P5','B3','B4','B5')
	virtual_gate_set_1.add_virtual_gates('vP4','vP5','vB3','vB4','vB5')
	virtual_gate_set_1.add_virtual_gate_matrix(np.eye(5))

	# # make virtual channels for IQ usage (also here, make one one of these object per MW source)
	# IQ_chan_set_1 = IQ_channel_constructor(pulse)
	# # set right association of the real channels with I/Q output.
	# IQ_chan_set_1.add_IQ_chan("I_MW", "I")
	# IQ_chan_set_1.add_IQ_chan("Q_MW", "Q")
	# IQ_chan_set_1.add_marker("M1", -15, 15)
	# IQ_chan_set_1.add_marker("M2", -15, 15)
	# # set LO frequency of the MW source. This can be changed troughout the experiments, bit only newly created segments will hold the latest value.
	# IQ_chan_set_1.set_LO(1e9)
	# # name virtual channels to be used.
	# IQ_chan_set_1.add_virtual_IQ_channel("MW_qubit_1")
	# IQ_chan_set_1.add_virtual_IQ_channel("MW_qubit_2")

	# finish initialisation (! important if using keysight uploader)
#	pulse.finish_init()

	return pulse

if __name__ == '__main__':
	pulse = return_pulse_lib()