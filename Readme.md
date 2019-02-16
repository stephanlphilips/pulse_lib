# Introduction

This is a pulse library that is build to make pulses that are commonly used to control spin qubits coherenly. A lot of attention is given to performance, structure and ease of use. At the moment the library only has a back-end that is suited for Keysight PXI AWG systems

Features now include:
* support for any pulse and sine waves (phase coherent atm)
* Sequencing
* Fully multidimensional. Execute any command as a loop in any dimension.
* Short and clean syntax. No sympy.
* Native support for virtual gates
* IQ toolkit and IQ virtual channels -- Full suppport for single sideband modulation (Along with PM/AM/FM)
* High speed uploader for Keysight PXI systems which supports upload during playback.

!! keysight AWG's their waveforms need to have a length of modulo 10 !! (related to the clock of the AWG)
--> segments are concatenated for this purose when uploading (e.g. upload happens in one big chunk)

# Requirements
You need python3.x and a c/c++ compiler. For the c-compiler, the following is recommended
* windows: the Visual Studio SDK C/C++ compiler (tested)
* linux: gcc is fine.
* ox x : gcc or clang both work

If you want to install the upload libraries for the keysight system, you will need:
* the Keysight SD1 software
* openMP (comes by default in visual studio)

# Quick start
The pulse library can be installed by cloning the library from github on your computer.
Navigate in the github folder and run the following in the terminal:
```bash
	python3 setup.py install
```
To install the uploading backend, you need to call:
```bash
	windows compilation under development
```

# Documentation
Documentation for the library can be found at:

https://qtwiki.tudelft.nl/pulse_lib

# Initializing the library

The general object that will manage all the segments and the interection with the AWG is the pulse object. The pulse object is best created in the station.
```python
	from pulse_lib.base_pulse import pulselib
	p = pulselib()
```

Let's intitialize the AWG's and make some channels in the pulse library
```python
	# init AWG
	awg1 = keysight_awg.SD_AWG('awg1', chassis = 0, slot= 2, channels = 4, triggers= 8)
	awg2 = keysight_awg.SD_AWG('awg2', chassis = 0, slot= 3, channels = 4, triggers= 8)
	awg3 = keysight_awg.SD_AWG('awg3', chassis = 0, slot= 4, channels = 4, triggers= 8)
	awg4 = keysight_awg.SD_AWG('awg4', chassis = 0, slot= 5, channels = 4, triggers= 8)
	
	# add to pulse_lib
	p.add_awgs('AWG1',awg1)
	p.add_awgs('AWG2',awg2)
	p.add_awgs('AWG3',awg3)
	p.add_awgs('AWG4',awg4)
	
	# define channels
	awg_channels_to_physical_locations = dict({'B0':('AWG1', 1), 'P1':('AWG1', 2),
		'B1':('AWG1', 3), 'P2':('AWG1', 4),
		'B2':('AWG2', 1), 'P3':('AWG2', 2),
		'B3':('AWG2', 3), 'P4':('AWG2', 4),
		'B4':('AWG3', 1), 'P5':('AWG3', 2),
		'B5':('AWG3', 3), 'G1':('AWG3', 4),
		'I_MW':('AWG4', 1), 'Q_MW':('AWG4', 2),	
		'M1':('AWG4', 3), 'M2':('AWG4', 4)})
		
	p.define_channels(awg_channels_to_physical_locations)
```
Now we have some channels, we can also specity delays (e.g. due to different lenght of coaxial cables, cables that pass through a mixer, ...).
```python
	# format : dict of channel name with delay in ns (can be posive/negative)
	p.add_channel_delay({'I_MW':50, 'Q_MW':50, 'M1':20, 'M2':-25, })
```

You might also want to define some virtual gates,
this can simply be done by defining:
```python
	awg_virtual_gates = {'virtual_gates_names_virt' :
		['vP1','vP2','vP3','vP4','vP5','vB0','vB1','vB2','vB3','vB4','vB5'],
				'virtual_gates_names_real' :
		['P1','P2','P3','P4','P5','B0','B1','B2','B3','B4','B5'],
		 'virtual_gate_matrix' : np.eye(11)
	}
	p.add_virtual_gates(awg_virtual_gates)
```
The matrix can be updated with (not yet implemented):
```python
	p.update_virtual_gate_matrix(my_matrix)
```
To generate virtual channels for IQ signals, you can add the following code:
```python
	awg_IQ_channels = {'vIQ_channels' : ['qubit_1','qubit_2'],
			'rIQ_channels' : [['I_MW','Q_MW'],['I_MW','Q_MW']],
			'LO_freq' :[MW_source.frequency, 1e9]
			# do not put the brackets for the MW source
			# e.g. MW_source.frequency
			}
	
	p.add_IQ_virt_channels(awg_IQ_channels)
```
Where the virtual channels are the new names of the new channels created to do IQ channals. Each of these channels refer to a real channel (note that you can refer to the same real channel multiple times). It is recomendable to create one IQ virtual channel for each qubit.

Note to finize the initialisation of the pulse library, you should call:
```python
	p.finish_init()
```
This will prepare all the segment/sequencing objects you might need later.

The code above can be placed in your station.

# Generating segments

Segments come by default in containers (e.g. for all the channels (real/virtual) you defined).
Note that all segments in these containers are assumed to have the same length (so P1, P2, P3, ...).

An example of defining some segments.
```python

	seg  = p.mk_segment('INIT')
	seg2 = p.mk_segment('Manip')
	seg3 = p.mk_segment('Readout')
```
Each segment has a unique name. This name we will use later for playback.

## Making your first pulse

To each segments you can add basic waveforms. Here you are free to add anything you want. 
Some examples follow (times are by default in ns).
```python
	# B0 is the barrier 0 channel
	# adds a linear ramp from 10 to 20 ns with amplitude of 5 to 10.
	seg.B0.add_pulse([[10.,0.],[10.,5.],[20.,10.],[20.,0.]])
	# add a block pulse of 2V from 40 to 70 ns, to whaterver waveform is already there
	seg.B0.add_block(40,70,2)
	# just waits (e.g. you want to ake a segment 50 ns longer)
	seg.B0.wait(50)
	# resets time back to zero in segment. Al the commannds we run before will be put at a negative time.
	seg.B0.reset_time()
	# this pulse will be placed directly after the wait()
	seg.B0.add_block(0,10,2)
```
# Playback of pulses

Once pulses are added, you can define a sequence like:
```python
	SEQ = [['INIT', 1, 0], ['Manip', 1, 0], ['Readout', 1, 0] ]
```
Where (e.g.) init is played once (1) and has a prescalor of 0 (see Keysight manual) 

This sequence can be added to the pulse object and next be uploaded.
```python
	p.add_sequence('mysequence', SEQ)
	p.start_sequence('mysequence')
```


# Some examples of easy pulses:
(more advanced examples follow later)
* 1D scan
```python
seg  = p.mk_segment('1D_scan')
#lets sweep the virtual plunger,
n_points = 100
swing = 1000 #mV
period = 1000 #ns

v_start =swing/2
step = swing/npoints

for i in range(n_points):
	seg.vP1.add_block(i*period, (i+1)period, v_start + i*step)
#done.
```
* 2D scan
```python
seg  = p.mk_segment('2D_scan')
#lets sweep the virtual plunger 1 versus virtual plunger 2,
n_points1 = 100
n_points2 = 100

swing1 = 1000 #mV
swing2 = 1000 #mV

period = 1000 #ns
period1 = period #ns
period2 = period*n_points1 #ns


v_start1 =swing1/2
step1 = swing1/npoints1
v_start1 =swing1/2
step1 = swing1/npoints1

for i in range(n_points1):
	seg.vP1.add_block(i*period1, (i+1)period1, v_start1 + i*step1)
seg.repeat(n_points2)
for i in range(n_points2):
	seg.vP1.add_block(i*period2, (i+1)period2, v_start2 + i*step2)
#done.
```
* 2D FM scan
```python
lets build on top of the perv sample. let att the modulation on the real plungers (1 and 2) and barrier (2).
seg.P1.add_sin(10e6, ...)
...

```
* Ramsey experiment (using loops)
```python
seg  = p.mk_segment('Ramsey')

# do pi/2 pulse
seg.IQ.add_sin(0,100e-9,freq = 2e9, amp = 10, phase = 0)
# wait -- use a loop object (has access to default numpy operators if it is numerical)
times = lp.linspace(5,20e3, 500, axis=0, name="time", unit="ns")
seg.IQ.wait(times)
# reset time
seg.IQ.reset_time()
# do pi/2 pulse
seg.IQ.add_sin(0,100e-9,freq = 2e9, amp = 10, phase = 0)
```
One can see that here a call to a single loopobject it made. We will be sweeping from 50ns to 20us. This sweep axis is 0 (first axis). The default is always a new axis. You only need to explicitely specify it if you want to parameters to be coupled (e.g. swept on the same axis). The name is optional and can be used for plotting, same for the unit.

* Rabi experiment (power vs frequency of the pulse): Good example of a two dimensioal sweep
```python
seg  = p.mk_segment('RABI')

t0 = 0
t1 = 50
freq = lp.linspace(1e9,1.1e9, 500, axis= 0, name="frequency", unit="Hz")
amp = lp.linspace(5,20e3, 500, axis= 1, name="VoltageIQ", unit="a.u.")
phase = 0
seg.IQ.add_sin(t0,t1,freq, amp, phase)
```

# TODO
TODO list:
* Update virtual gate matrix function
* Support for calibarion arguments? -- this should be engineered well.
* HVI2 integration
* add DSP module (must also be C++)

TODO bugs and small things to fix,
* deal with names and units of the loops + setpoints variable
