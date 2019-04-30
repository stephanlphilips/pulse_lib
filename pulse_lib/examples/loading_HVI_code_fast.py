"""
make a pulse object
"""


"""
define a function that loads the HVI file that will be used thoughout the experiments
"""
import keysightSD1

def load_HVI(AWGs, channel_map):
	"""
	load a HVI file on the AWG.
	Args:
		AWGS (dict <str, QCoDeS Intrument>) : key is AWGname, value awg object. 
		channel_map (dict <str, (tuple <str, int>)) : key is channelname, value is AWGname, channel number
	Returns:
		HVI (SD_HVI) : keyisight HVI object.	
	"""
	for channel, channel_loc in channel_map.items():
		# 6 is the magic number of the arbitary waveform shape.
		AWGs[channel_loc[0]].awg_stop(channel_loc[1])
		AWGs[channel_loc[0]].set_channel_wave_shape(keysightSD1.SD_Waveshapes.AOU_AWG,channel_loc[1])
		AWGs[channel_loc[0]].awg_queue_config(channel_loc[1], 1)

			
	HVI = keysightSD1.SD_HVI()
	HVI.open("C:/V2_code/HVI/HVI_playback_restless.HVI")

	HVI.assignHardwareWithUserNameAndSlot("Module 0",0,2)
	HVI.assignHardwareWithUserNameAndSlot("Module 1",0,3)
	HVI.assignHardwareWithUserNameAndSlot("Module 2",0,4)
	HVI.assignHardwareWithUserNameAndSlot("Module 3",0,5)

	HVI.compile()
	HVI.load()
	

	HVI.start()

	return HVI


"""
define a function that applies the settings to a HVI file and then compiles it before the experiment.
"""

def set_and_compile_HVI(HVI, playback_time, n_rep, *args, **kwargs):
	"""
	Function that set values to the currently loaded HVI script and then performs a compile step.
	Args:
		HVI (SD_HVI) : HVI object that is already loaded in the memory. Will be loaded by default.
		playback_time (int) : #ns to play the sequence (assuming every point is one ns)
		n_rep (int) : number of repertitions. This is the number of reperititons that you set in the pulselub object.
	Returns:
		None
	"""
	# No need ... We will overwrite the registers instead of a re-compile for updated speed :-)
	pass

"""
Function to load the HVI on the AWG. This will be the last function that is executed in the play function.

This function is optional, if not defined, there will be just two calls,
	HVI.load()
	HVI.start()
So only define if you want to set custom settings just before the experiment starts. Note that you can access most settings via HVI itselves, so it is better to do it via there.
"""

def excute_HVI(HVI, AWGs, channel_map, playback_time, n_rep, *args, **kwargs):
	"""
	load HVI code.
	Args:
		AWGS (dict <str, QCoDeS Intrument>) : key is AWGname, value awg object. 
		channel_map (dict <str, (tuple <str, int>)) : key is channelname, value is AWGname, channel number
		playback_time (int) : #ns to play the sequence (assuming every point is one ns)
		n_rep (int) : number of repertitions. This is the number of reperititons that you set in the pulselub object.
	"""

	nrep = int(n_rep)
	step = 1

	length = int(playback_time/10 + 20)

	for awgname, awg in AWGs.items():
		awg.writeRegisterByNumber(2, int(nrep))
		awg.writeRegisterByNumber(3, int(length))

	# writing a 1 to the register starts the AWG.
	AWGs['AWG1'].writeRegisterByNumber(0,int(1))

if __name__ == '__main__':
	
	"""
	Let's now set these setting to the AWG, for this peculiar experiment.
	"""
	pulse = return_pulse_lib()

	test  = pulse.mk_segment()
	test.P1 += 100
	test.P1.wait(1e6)

	my_seq = pulse.mk_sequence([test])

	# set number of repetitions (default is 1000)
	my_seq.n_rep = 10000
	my_seq.add_HVI(load_HVI, set_and_compile_HVI, excute_HVI)

	# my_seq.upload(0)
	# my_seq.start(0)
	# # start upload after start.
	# my_seq.upload(0)
	# # upload is released shortly after called (~5ms, upload is without any gil.)

