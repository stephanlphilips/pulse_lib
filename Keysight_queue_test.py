# Note that qcodes is used (just a wrapper for the keysight driver)

import qcodes.instrument_drivers.Keysight.SD_common.SD_AWG as keysight_awg
import qcodes.instrument_drivers.Keysight.SD_common.SD_DIG as keysight_dig

awg1 = keysight_awg.SD_AWG('awg1', chassis = 0, slot= 2, channels = 4, triggers= 8)
awg2 = keysight_awg.SD_AWG('awg2', chassis = 0, slot= 3, channels = 4, triggers= 8)
awg3 = keysight_awg.SD_AWG('awg3', chassis = 0, slot= 4, channels = 4, triggers= 8)
awg4 = keysight_awg.SD_AWG('awg4', chassis = 0, slot= 5, channels = 4, triggers= 8)
dig1 = keysight_dig.SD_DIG('dig1', chassis = 0, slot= 6, channels = 4, triggers= 8)

import time
import numpy as np

awg1.awg_stop(1)
awg2.awg_stop(1)
awg3.awg_stop(1)
awg4.awg_stop(1)

awg1.awg_stop(2)
awg2.awg_stop(2)
awg3.awg_stop(2)
awg4.awg_stop(2)


awg1.awg_stop(3)
awg2.awg_stop(3)
awg3.awg_stop(3)
awg4.awg_stop(3)

awg1.awg_stop(4)
awg2.awg_stop(4)
awg3.awg_stop(4)
awg4.awg_stop(4)

# print(awg1.awg.clockResetPhase(3,4004,0))
# print(awg2.awg.clockResetPhase(3,4004,0))
# print(awg3.awg.clockResetPhase(3,4004,0))
# print(awg4.awg.clockResetPhase(3,4004,0))

# # awg1.reset_clock_phase(3,4004)
# # awg2.reset_clock_phase(3,4004)
# # awg3.reset_clock_phase(3,4004)
# # awg4.reset_clock_phase(3,4004)


# awg1.awg.PXItriggerWrite(4004,1)
# awg1.awg.PXItriggerWrite(4004,0)
# awg1.awg.PXItriggerWrite(4004,1)
# awg1.reset_channel_phase(1, True)
# awg1.reset_channel_phase(2, True)
# awg1.awg.clockSetFrequency()

# raise
# awg1.awg_stop(1)
awg1.awg_flush(1)
awg1.awg_flush(2)

awg1.flush_waveform()

# awg2.awg_stop(1)
awg2.awg_flush(1)
awg2.flush_waveform()

awg3.awg_flush(1)
awg3.flush_waveform()

a = np.linspace(0,0.5,1000, dtype=np.double)
a[0:10] = 1
b = np.linspace(-0.5,0.5,2000, dtype=np.double)

b[-1] = -1

# awg1.set_channel_frequency(0.5e6/3,1)
# awg1.set_channel_frequency(0.5e6/3,2)
# awg2.set_channel_frequency(0.5e6/3,1)
# awg4.set_channel_frequency(0.5e6/3,4)
# awg1.set_channel_wave_shape(6,1)
# awg1.set_channel_wave_shape(4,2)
# awg2.set_channel_wave_shape(6,1)
# awg4.set_channel_wave_shape(4,4)

amp = 1
off = 0
awg1.set_channel_amplitude(amp,1)
awg1.set_channel_offset(off,1)
awg1.set_channel_amplitude(amp,2)
awg1.set_channel_offset(off,2)
awg2.set_channel_amplitude(amp,1)
awg2.set_channel_offset(off,1)
awg3.set_channel_amplitude(amp,1)
awg3.set_channel_offset(off,1)

w1 = keysight_awg.SD_AWG.new_waveform_from_double(0, a)
w2 = keysight_awg.SD_AWG.new_waveform_from_double(0, b)

awg1.load_waveform(w1, 1)
awg1.load_waveform(w2, 2)

awg2.load_waveform(w1, 1)
awg2.load_waveform(w2, 2)

awg3.load_waveform(w1, 1)
awg3.load_waveform(w2, 2)

awg1.awg_queue_waveform(1,1,1,0,1,0)
awg1.awg_queue_waveform(1,2,0,0,1,0)

awg1.awg_queue_waveform(2,1,1,0,1,0)
awg1.awg_queue_waveform(2,2,0,0,1,0)

awg2.awg_queue_waveform(1,1,1,0,1,0)
awg2.awg_queue_waveform(1,2,0,0,1,0)

awg3.awg_queue_waveform(1,1,1,0,1,0)
awg3.awg_queue_waveform(1,2,0,0,1,0)
# awg1.awg_queue_waveform(1,1,0,0,2,0)
# awg1.awg_queue_waveform(1,2,0,0,2,0)
# awg1.awg_queue_waveform(1,1,0,0,2,0)
# awg1.awg_queue_waveform(1,2,0,0,2,0)

# awg1.awg.AWGqueueSyncMode(1,1)
# awg1.awg.AWGqueueSyncMode(2,1)
# awg2.awg.AWGqueueSyncMode(1,1)
# awg3.awg.AWGqueueSyncMode(1,1)

# awg1.awg_config_external_trigger(1,4005,1)
# awg1.awg_config_external_trigger(2,4005,1)
# awg2.awg_config_external_trigger(1,4005,1)
# awg3.awg_config_external_trigger(1,4005,1)

# awg1.awg_queue_config(1, 1)
# awg1.awg_queue_config(2, 1)
# awg2.awg_queue_config(1, 1)
# awg3.awg_queue_config(1, 1)



# awg3.awg.PXItriggerWrite(4005,0)

awg1.awg_start(1)
awg1.awg_start(2)
awg2.awg_start(1)
awg3.awg_start(1)
# awg3.awg.PXItriggerWrite(4005,1)
# awg3.awg.PXItriggerWrite(4005,0)
# awg3.awg.PXItriggerWrite(4005,1)
# awg3.awg.PXItriggerWrite(4005,0)
# awg3.awg.PXItriggerWrite(4005,1)
# awg3.awg.PXItriggerWrite(4005,0)
# awg3.awg.PXItriggerWrite(4005,1)
# awg3.awg.PXItriggerWrite(4005,0)
# time.sleep(20)


awg1.awg.close()
awg2.awg.close()
awg3.awg.close()
awg3.awg.close()
