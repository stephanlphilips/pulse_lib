# Introduction

This is a simple pulse library that is made to work together with the Keysight AWG. The main motivation for this library was to have a very easy and structured way to make waveforms. Also attention is given to performance for the construction of the waveform. The construction should always be faster than upload time (this was challaging since the upload goes via pcie + python ..). Performance critical parts are written in cython. Note that it is not the intention to support other AWG systems with this project (though the pulse builder should be genereric).

Features now include:
* Native support for virtual gates
* support for any pulse and sine waves (phase coherent atm)
* Sequencing
* delay in awg lines.
* Fully multidimensionlisation of the segment object (with matrix operators) ==> advanced (integrated) looping methods -- decorator approach + looping class. 
* IQ toolkit and IQ virtual channels -- Full suppport for single sideband modulation (in combination with phase/amp/frequency modulation)

	
TODO list:
* Update virtual gate matrix function
* more base functions
* Support for calibarion arguments? -- this should be engineered well.
* Memory segmentation on the keysight awg. This will be needed if you want to upload during an experiment.
* Faster add function for block funtion (now performace issues if more than ~2000 elements in a sequence (not nice to call a lot)).

# Initializing the library

The general object that will manage all the segments and the interection with the AWG is the pulse object. The pulse object is best created in the station.
```python
	p = pulselib()
```
You might also want to define some virtual gates,
this can simply be done by defining:
```python
	awg_virtual_channels = {'virtual_gates_names_virt' :
		['vP1','vP2','vP3','vP4','vP5','vB0','vB1','vB2','vB3','vB4','vB5'],
				'virtual_gates_names_real' :
		['P1','P2','P3','P4','P5','B0','B1','B2','B3','B4','B5'],
		 'virtual_gate_matrix' : np.eye(11)
	}
	p.add_virtual_gates(awg_virtual_gates)
```
The matrix can be updated with:
```python
	p.update_virtual_gate_matrix(my_matrix)
```
To generate virtual channels for IQ signals, you can add the following code:
```python
	awg_IQ_channels = {'vIQ_channels' : ['IQ1','IQ2'],
			'rIQ_channels'	[['I1','Q1']['I2','Q2']],
			'LO_freq' :[MW_source.frequency, 1e9]
			# do not put the brackets for the MW source
			}
	p.add_IQ_virt_channels(awg_IQ_channels)
```
Where the virtual channels are the new names of the new channels created to do IQ channals. Each of these channels refer to a real channel (note that you can refer to the same real channel multiple times). It is recomendable to create one IQ virtual channel for each qubit.

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
	seg.B0.add_pulse([[10,0],[10,5],[20,10],[20,0]]) # adds a linear ramp from 10 to 20 ns with amplitude of 5 to 10.
	# B0 is the barrier 0 channel
	seg.B0.add_block(40,70,2) # add a block pulse of 2V from 40 to 70 ns, to whaterver waveform is already there
	seg.B0.wait(50) # just waits (e.g. you want to ake a segment 50 ns longer)
	seg.B0.reset_time(). # resets time back to zero in segment. Al the commannds we run before will be put at a negative time.
	seg.B0.add_block(0,10,2) # this pulse will be placed directly after the wait()
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

# working with calibarated elemements.
