import uploader

t = uploader.keysight_upload_module()
# t.add_awg_module("AWG1", 0 , 1)
import numpy as np
up = uploader.waveform_upload_chache((-0.1,0.1))
up.add_data(np.linspace(0,10,5000), (0.,10.), 1000.)
up.add_data(np.linspace(0,10,5000), (0.,10.), 1000.)

print(up.integral)
print(up.compensation_time)

awg_channels_to_physical_locations = dict({'B0':('AWG1', 1), 'P1':('AWG1', 2),
		'B1':('AWG1', 3), 'P2':('AWG1', 4),
		'B2':('AWG2', 1), 'P3':('AWG2', 2),
		'B3':('AWG2', 3), 'P4':('AWG2', 4),
		'B4':('AWG3', 1), 'P5':('AWG3', 2),
		'B5':('AWG3', 3), 'G1':('AWG3', 4),
		'I_MW':('AWG4', 1), 'Q_MW':('AWG4', 2),	
		'M1':('AWG4', 3), 'M2':('AWG4', 4)})

c = uploader.waveform_cache_container(awg_channels_to_physical_locations)
for key, val in awg_channels_to_physical_locations.items():
	c[key] = up

print(c.npt)
# t.add_upload_data(c)