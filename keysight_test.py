import qcodes.instrument_drivers.Keysight.SD_common.SD_AWG as keysight_awg
import qcodes.instrument_drivers.Keysight.SD_common.SD_DIG as keysight_dig

awg1 = keysight_awg.SD_AWG('awg1', chassis = 0, slot= 2, channels = 4, triggers= 8)
awg2 = keysight_awg.SD_AWG('awg2', chassis = 0, slot= 3, channels = 4, triggers= 8)
awg3 = keysight_awg.SD_AWG('awg3', chassis = 0, slot= 4, channels = 4, triggers= 8)
awg4 = keysight_awg.SD_AWG('awg4', chassis = 0, slot= 5, channels = 4, triggers= 8)

awg1.set_channel_frequency(50e6,1,True)
awg1.set_channel_frequency(50e6,2,True)
awg2.set_channel_frequency(50e6,1,True)

# set_channel_phase
awg1.set_channel_amplitude(1,1)
# set_channel_offset
awg1.set_channel_wave_shape(4,1)

# load_waveform
# load_waveform_int16
# reload_waveform_int16

# flush_waveform
# awg_queue_waveform
# awg_queue_config
# awg_flush
# awg_start
# awg_start_multiple

# awg_stop_multiple

# awg_config_external_trigger

# import base_pulse
# p = base_pulse.pulselib()
# p.add_awgs([awg1,awg2,awg3,awg4])
# seg = p.mk_segment('test')
# # append functions?
# seg.B0.add_pulse([[10,5]
# 				 ,[20,5]])

# seg.B0.add_pulse([[20,0],[30,5], [30,0]])
# seg.B0.add_block(40,70,2)
# seg.B0.add_pulse([[70,0],
# 				 [80,0],
# 				 [150,5],
# 				 [150,0]])
# # seg.B0.repeat(20)
# # seg.B0.wait(20)
# # print(seg.B0.my_pulse_data)
# # seg.reset_time()
# seg.B1.add_pulse([[10,0],
# 				[10,5],
# 				[20,5],
# 				[20,0]])
# seg.B1.add_block(20,50,2)

# seg.B1.add_block(80,90,2)
# seg.B1.plot_sequence()