from pulse_lib.base_pulse import pulselib
from pulse_lib.virtual_channel_constructors import IQ_channel_constructor, virtual_gates_constructor

import qcodes.instrument_drivers.Keysight.SD_common.SD_AWG as keysight_awg
import qcodes.instrument_drivers.Keysight.SD_common.SD_DIG as keysight_dig
import numpy as np

def return_pulse_lib():
	pulse = pulselib()

	AWG1 = keysight_awg.SD_AWG('my_awg1', chassis = 0, slot= 2, channels = 4, triggers= 8)
	AWG2 = keysight_awg.SD_AWG('my_awg2', chassis = 0, slot= 3, channels = 4, triggers= 8)
	AWG3 = keysight_awg.SD_AWG('my_awg3', chassis = 0, slot= 4, channels = 4, triggers= 8)
	AWG4 = keysight_awg.SD_AWG('my_awg4', chassis = 0, slot= 5, channels = 4, triggers= 8)

	# add to pulse_lib
	pulse.add_awgs('AWG1',AWG1)
	pulse.add_awgs('AWG2',AWG2)
	pulse.add_awgs('AWG3',AWG3)
	pulse.add_awgs('AWG4',AWG4)

	# define channels
	# pulse.define_channel('B0','AWG1', 1)
	# pulse.define_channel('P1','AWG1', 2)
	# pulse.define_channel('B1','AWG1', 3)
	# pulse.define_channel('P2','AWG1', 4)
	pulse.define_channel('B2','AWG2', 1)
	pulse.define_channel('P3','AWG2', 2)
	pulse.define_channel('B3','AWG2', 3)
	pulse.define_channel('P4','AWG2', 4)
	pulse.define_channel('I_MW','AWG4',1)
	pulse.define_channel('Q_MW','AWG4',2)
	pulse.define_marker('M1','AWG4', 3)
	pulse.define_marker('M2','AWG4', 4)
	pulse.define_channel('B4','AWG3', 1)
	pulse.define_channel('P5','AWG3', 2)
	pulse.define_channel('B5','AWG3', 3)
	pulse.define_channel('G1','AWG3', 4)


	# format : channel name with delay in ns (can be posive/negative)
	pulse.add_channel_delay('I_MW',50)
	pulse.add_channel_delay('Q_MW',50)
	pulse.add_channel_delay('M1',20)
	pulse.add_channel_delay('M2',-25)

	# add limits on voltages for DC channel compenstation (if no limit is specified, no compensation is performed).
	pulse.add_channel_compenstation_limit('B4', (-500, 500))
	pulse.add_channel_compenstation_limit('P5', (-500, 500))

	# set a virtual gate matrix (note that you are not limited to one matrix if you would which so)
	# virtual_gate_set_1 = virtual_gates_constructor(pulse)
	# virtual_gate_set_1.add_real_gates('P1','P2','P3','P4','P5','B0','B1','B2','B3','B4','B5')
	# virtual_gate_set_1.add_virtual_gates('vP1','vP2','vP3','vP4','vP5','vB0','vB1','vB2','vB3','vB4','vB5')
	# virtual_gate_set_1.add_virtual_gate_matrix(np.eye(11))

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
	pulse.finish_init()

	return pulse

if __name__ == '__main__':
	pulse = return_pulse_lib()