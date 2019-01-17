import uploader
import time 


class AWG(object):
	"""docstring for AWG"""
	def __init__(self, name):
		self.name = name
		self.chassis = 0
		self.slot = 0
		self.type = "DEMO"

AWG1 = AWG("AWG1")
AWG2 = AWG("AWG2")
AWG3 = AWG("AWG3")
AWG4 = AWG("AWG4")
		


t = uploader.keysight_upload_module()
t.add_awg_module("AWG1",AWG1)
t.add_awg_module("AWG2",AWG2)
t.add_awg_module("AWG3",AWG3)
t.add_awg_module("AWG4",AWG4)

import numpy as np
up = uploader.waveform_upload_chache((-0.1,0.1))
wvf =np.linspace(10,10.,10000, dtype=np.double)
up.add_data(wvf, (0.,10.), 1000.)
up.add_data(np.linspace(0,10,5000), (0.,10.), 1000000.)


awg_channels_to_physical_locations = dict({'B0':('AWG1', 1), 'P1':('AWG1', 2), 
'B1':('AWG1', 3), 'P2':('AWG1', 4),
		'B2':('AWG2', 1), 'P3':('AWG2', 2),
		'B3':('AWG2', 3), 'P4':('AWG2', 4),
		'B4':('AWG3', 1), 'P5':('AWG3', 2),
		'B5':('AWG3', 3), 'G1':('AWG3', 4),
		'I_MW':('AWG4', 1), 'Q_MW':('AWG4', 2),	
		'M1':('AWG4', 3), 'M2':('AWG4', 4)})

min_max_voltages_compenstaion = dict({'B0': (-1.,1), 'P1':(-1,1),
											'B1': (-1.,1), 'P2':(-1,1),
											'B2': (-1.,1), 'P3':(-1,1),
											'B3': (-1.,1), 'P4':(-1,1),
											'B4': (-1.,1), 'P5':(-1,1),
											'B5': (-1.,1), 'G1':(-1,1),
											'I_MW': (-1.,1), 'Q_MW':(-1,1),	
											'M1': (-1.,1), 'M2':(-1,1)})

c = uploader.waveform_cache_container(awg_channels_to_physical_locations, min_max_voltages_compenstaion)

for key in awg_channels_to_physical_locations:
    c[key].add_data(np.linspace(0,10), (0,10),50)

print("npt",c[key].npt)
s =time.time()
t.add_upload_data(c)
stop = time.time()
print("time eplaced = {}".format(stop-s))

# def rescale(wfv, Vmin, Vmax):
# 	voff = (Vmax  + Vmin)/2
# 	vpp  = (Vmax - Vmin)
# 	bitscale = 65535
# 	single_bit_step = 0.5/65535
# 	offset_factor = voff + single_bit_step*vpp
# 	rescaling_factor = bitscale/vpp

# 	return (wfv - offset_factor)*rescaling_factor

# print(rescale(np.linspace(0,1,50), 0,1))



