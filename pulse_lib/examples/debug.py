import numpy as np

from pulse_lib.base_pulse import pulselib
from example_init import return_pulse_lib
from charge_stability_diagram import construct_ct
# from loading_HVI_code_fast_single_shot import load_HVI, set_and_compile_HVI, excute_HVI
from loading_HVI_code_fast import load_HVI, set_and_compile_HVI, excute_HVI


import qcodes.instrument_drivers.Keysight.SD_common.SD_DIG_new_firmware as keysight_dig
dig = keysight_dig.SD_DIG(name ="keysight_digitizer", chassis = 0, slot = 6, channels = 4, triggers=1)

dig.daq_flush(1)
dig.daq_flush(2)
dig.daq_flush(3)
dig.daq_flush(4)

def set_digitizer(digitizer1, cycles = 10, Vmax = 2):
    digitizer1.channels.ch1.set_channel_properties(Vmax)
    digitizer1.channels.ch2.set_channel_properties(Vmax)

    digitizer1.channels.ch1.set_daq_settings(cycles)
    digitizer1.channels.ch2.set_daq_settings(cycles)

set_digitizer(dig)

pulse, virtual_gate = return_pulse_lib()
# pulse.cpp_uploader.resegment_memory()

test = pulse.mk_segment()

test.A4.add_marker(0,10)

test.A4.add_marker(30,50)
test.A4.wait(5000)

gate1 = "A6"
n_pt = 50
vpp = 1000
t_step = 500
for  voltage in np.linspace(-vpp,vpp,n_pt):
	test.A6.add_block(0, t_step, voltage)
	test.A6.reset_time()
test.A6.repeat(n_pt)
test.A6.reset_time()
test.A6.add_block(0, t_step, 1100)
n_pt = 3
t_step = 50
#test.A2.add_block(0, t_step, 50)
#test.A2.reset_time()
test.A2.add_block(0, 500, -250)
test.A2.wait(500)
test.A2.reset_time()
test.A2.add_block(0, 500, 250)
test.A2.wait(2000)

sequence = [test]
# sequence = [construct_ct("A6", "A2", "A4",1000 ,1000, 30)]
my_seq = pulse.mk_sequence(sequence)
my_seq.add_HVI(load_HVI, set_and_compile_HVI, excute_HVI)#, digitizer = dig, dig_wait = 100)
my_seq.n_rep = 10

my_seq.neutralize = True

print("uploading starts")
my_seq.upload([0])
print("play is starting")
my_seq.play([0])

# pulse.uploader.wait_until_AWG_idle()
# pulse.uploader.release_memory()
