# Me wrinting a minimalistic readme

This is just a quick and dirty implementation of a pulse library for the keysight modules. 
I was not really in the mood to read in the 2000 + lines of code of the other libraries so I decied to maka a quick one taylored to the keysight awg's

The libraries are quite simple, in the sense that that you just start from a pulse lib object 

e.g.
```python

	p = pulselib()
```

here you can define segments:: 
```python

	seg  = p.mk_segment('INIT')
	seg2 = p.mk_segment('Manip')
	seg3 = p.mk_segment('Readout')
```

Then to each segments you can add some basic waveform. Here you are free to add anything you want. 
Some examples follow (times are by default in ns).
Note that here is chosen for a time based approach rather than the conventional stick stuff after each other approach. I my opnion in this way it should be easier to keep oversight.
```python
	seg.B0.add_pulse([[10,5]
					 ,[20,5]])
	# B0 is the barrier 0 channel
	seg.B0.add_pulse([[20,0],[30,5], [30,0]])
	seg.B0.add_block(40,70,2)
	seg.B0.add_pulse([[70,0],
					 [80,0],
					 [150,5],
					 [150,0]])
```

One pulses are added, you can define like:
```python
	SEQ = [['INIT', 1, 0], ['Manip', 1, 0], ['Readout', 1, 0] ]
```

Which can next be added to the pulse object and next be uploaded.
```python
	p.add_sequence('mysequence', SEQ)

	p.start_sequence('mysequence')
```
Virtual gates are also supported. This can simply be done by definig:
```python
	awg_virtual_channels = {'virtual_gates_names_virt' : ['vP1','vP2','vP3','vP4','vP5','vB0','vB1','vB2','vB3','vB4','vB5'],
									 'virtual_gates_names_real' : ['P1','P2','P3','P4','P5','B0','B1','B2','B3','B4','B5'],
									 'virtual_gate_matrix' : np.eye(11)}
```
Note that thresolds are chosen automatically. Memory menagment is also taken care of for you  :)


Features needed: 
* auto prescale -- only produce full numpy arrays when needed. 
* faster add function for block funtion (now performace issues of more then ~2000 elements in a sequence (not nice to call a lot)).
* advanced (integrated) looping methods -- decorator approach + looping class. Support for calibarion arguments? -- this should be enegnneerd well.
* more base functions
	* e.g. (IQ toolkit and IQ virtual channels) -- IF IQ is key for high performance (for IQ offset).
	* Normal channels phase coherennt or not??
* support for memory segmentation for the keysight -- upload during experiment capelibility
