from example_init import return_pulse_lib

pulse = return_pulse_lib()
my_new_pulse_container  = pulse.mk_segment()
print(my_new_pulse_container.channels)

start = 10 # ns
stop = 50 # ns
amp = 100 # mV
my_new_pulse_container.P1.add_block(start, stop, amp)