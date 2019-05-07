from pulse_lib.base_pulse import pulselib
import qcodes.instrument_drivers.Keysight.SD_common.SD_AWG as keysight_awg
import qcodes.instrument_drivers.Keysight.SD_common.SD_DIG as keysight_dig

import numpy as np
import time

# awg1 = keysight_awg.SD_AWG('my_awg1', chassis = 0, slot= 2, channels = 4, triggers= 8)
# awg2 = keysight_awg.SD_AWG('my_awg2', chassis = 0, slot= 3, channels = 4, triggers= 8)


# # ar = np.zeros([5000])
# # ar[:1000] = 1

# # awg1.waveformFlush()
# # awg.AWGflush(channel)
# awg1.set_channel_wave_shape(6,1)
# awg1.set_channel_amplitude(0.8, 1)
# awg1.set_channel_offset(0, 1)

# # wfv = awg1.wave.newFromArrayInteger(0,(ar*32767).astype(np.int))
# # wfv = keysight_awg.SD_AWG.new_waveform_from_double(0, ar.tolist())

# # awg1.load_waveform(wfv, 0)


# awg1.awg_queue_waveform(1,0,1,0,100000,0)
# awg1.awg.setDigitalFilterMode(2)
# awg1.awg_start(1)
# awg1.awg_trigger(1)

# time.sleep(3)

# awg1.awg_stop(1)
# awg1.awg_queue_waveform(1,44,1,0,100000,0)
# awg1.awg.setDigitalFilterMode(2)
# awg1.awg_start(1)
# awg1.awg_trigger(1)

# time.sleep(3)

# awg1.awg_stop(1)


# ----------
# Python - Sample Application to set up the AWG
# to output an array that was created with numpy.
# ----------
# Import required system components
import sys
# ----------
# Append the system path to include the
# location of Keysight SD1 Programming Libraries then import the library
import keysightSD1 # Import Python SD1 library and AWG/Digitizer commands.
import numpy # Import numpy which is used to make an array
# ----------
# Specify values for variables
product = 'M3202A' # Product's model number
chassis = 0 # Chassis number holding product
slot = 4 # Slot number of product in chassis
channel = 2 # Channel being used
amplitude = 1 # (Unit: Vp) Amplitude of AWG output signal (0.1 Vp)
waveshape = keysightSD1.SD_Waveshapes.AOU_AWG # Specify AWG output
delay = 0 # (Unit: ns) Delay after trigger before generating output.
cycles = 0 # Number of cycles. Zero specifies infinite cycles.
              # Otherwise, a new trigger is required to actuate each cycle
prescaler = 0 # Integer division reduces high freq signals to lower frequency
# ----------
# Select settings and use specified variables
awg = keysightSD1.SD_AOU() # Creates SD_AOU object called awg
awg.openWithSlot(product, chassis, slot) # Connects awg object to module
awg.channelAmplitude(channel, amplitude) # Sets output amplitude for awg
awg.channelWaveShape(channel, waveshape) # Sets output signal type for awg
awg.waveformFlush() # Cleans the queue
awg.AWGflush(channel) # Stops signal from outputing out of channel 1
# Create an array that represents a sawtooth signal using "numpy"
array = numpy.zeros(10000) # Create array of zeros with 1000 elements
array[0] = -1 # Initialize element 0 as -0.5
for i in range(1, len(array)): # This for..loop will increment from -0.5
	array[i] = array[i-1] + .0002 # Increment by .001 every iteration
wave = keysightSD1.SD_Wave() # Create SD_Wave object and call it "wave"
# (will place the array inside "wave")
error = wave.newFromArrayDouble(keysightSD1.SD_WaveformTypes.WAVE_ANALOG,
array.tolist()) # Place the array into the "wave" object
waveID = 0 # This number is arbitrary and used to identify the waveform
awg.waveformLoad(wave, waveID) # Load the "wave" object and give it an ID
awg.AWGqueueWaveform(channel, waveID, keysightSD1.SD_TriggerModes.SWHVITRIG,
delay, cycles, prescaler) # Queue waveform to prepare it to be output
# ----------
awg.AWGstart(channel) # Start the AWG
awg.AWGtrigger(channel) # Trigger the AWG to begin
# ----------
# Close the connection between the AWG object and the physical AWG hardware.
awg.close()
# ----------
# © Keysight Technologies, 2018
# All rights reserved.
# You have a royalty-free right to use, modify, reproduce
# and distribute this Sample Application (and/or any modified
# version) in any way you find useful, provided that
# you agree that Keysight Technologies has no warranty,
# obligations or liability for any Sample Application Files.
#
# Keysight Technologies provides programming examples
# for illustration only. This sample program assumes that
# you are familiar with the programming language being
# demonstrated and the tools used to create and debug
# procedures. Keysight Technologies support engineers can
# help explain the functionality of Keysight Technologies
# software components and associated commands, but they
# will not modify these samples to provide added
# functionality or construct procedures to meet your
# specific needs.
# ----------