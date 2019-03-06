"""
make a pulse object
"""
from example_init import return_pulse_lib

pulse = return_pulse_lib()


"""
define a function that loads the HVI file that will be used thoughout the experiments
"""
def load_HVI():
	"""
	load a HVI file on the AWG.
	Args:
		None
	Returns:
		HVI (SD_HVI) : keyisight HVI object.	
	"""
	HVI = keysightSD1.SD_HVI()
	HVI.open("C:/V2_code/HVI/For_loop_single_sequence.HVI")

	HVI.assignHardwareWithUserNameAndSlot("Module 0",0,2)
	HVI.assignHardwareWithUserNameAndSlot("Module 1",0,3)
	HVI.assignHardwareWithUserNameAndSlot("Module 2",0,4)
	HVI.assignHardwareWithUserNameAndSlot("Module 3",0,5)

	return HVI


"""
define a function that applies the settings to a HVI file and then compiles it before the experiment.
"""

def set_and_compile_HVI(HVI, npt, n_rep, *args, **kwargs):
	"""
	Function that set values to the currently loaded HVI script and then performs a compile step.
	Args:
		HVI (SD_HVI) : HVI object that is already loaded in the memory. Will be loaded by default.
		npt (int) : number of points of the sequence (assuming every point is one ns) (this is a default argument that will be provided by the pulse_lib library)
		n_rep (int) : number of repertitions. This is the number of reperititons that you set in the pulselub object.
	Returns:
		None
	"""

	# Length of the sequence
	HVI.writeIntegerConstantWithIndex(0, "length_sequence", int(npt/10 + 20))
	HVI.writeIntegerConstantWithIndex(1, "length_sequence", int(npt/10 + 20))
	HVI.writeIntegerConstantWithIndex(2, "length_sequence", int(npt/10 + 20))
	HVI.writeIntegerConstantWithIndex(3, "length_sequence", int(npt/10 + 20))


	# number of repetitions
	nrep = n_rep
	if nrep == 0:
		nrep = 1

	HVI.writeIntegerConstantWithIndex(0, "n_rep", nrep)
	HVI.writeIntegerConstantWithIndex(1, "n_rep", nrep)
	HVI.writeIntegerConstantWithIndex(2, "n_rep", nrep)
	HVI.writeIntegerConstantWithIndex(3, "n_rep", nrep)

	# Inifinite looping
	step = 1
	if n_rep == 0:
		step  = 0
	HVI.writeIntegerConstantWithIndex(0, "step", step)
	HVI.writeIntegerConstantWithIndex(1, "step", step)
	HVI.writeIntegerConstantWithIndex(2, "step", step)
	HVI.writeIntegerConstantWithIndex(3, "step", step)

	HVI.compile()

"""
Function to load the HVI on the AWG. This will be the last function that is executed in the play function.

This function is optional, if not defined, there will be just two calls,
	HVI.load()
	HVI.start()
So only define if you want to set custom settings just before the experiment starts. Note that you can access most settings via HVI itselves, so it is better to do it via there.
"""

def load_HVI():
	# TODO define AWGs..
	for awg in awgs:
		# set a certain filter on the FPGA
		awg.setDigitalFilterMode(2)
	
	HVI.load()
	HVI.start()


"""
Let's now set these setting to the AWG, for this peculiar experiment.
"""

my_seq = p.mk_sequence(sequence)

# set number of repetitions (default is 1000)
my_seq.n_rep = 10000
my_seq.add_HVI(load_HVI, set_and_compile_HVI, load_HVI)

my_seq.upload(0)
my_seq.start(0)
# start upload after start.
my_seq.upload(0)
# upload is released shortly after called (~5ms, upload is without any gil.)

