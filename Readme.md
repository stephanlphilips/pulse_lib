= Me wrinting a minimalistic readme =

This is just a quick and dirty implementation of a pulse library for the keysight modules. 
I was not really in the mood to read in the 2000 + lines of code of the other libraries so I decied to maka a quick one taylored to the keysiht awg's

The libraries are quite simple, in the sense that that you just start from a pulse lib object 

e.g.

	p = pulselib()

here you can define segments:: 

	seg  = p.mk_segment('INIT')
	seg2 = p.mk_segment('Manip')
	seg3 = p.mk_segment('Readout')

Then to each segments you can add some basic waveform. Here you are free to add anything you want. 
Some examples follow (times are by default in ns).
Note that here is chosen for a time based approach rather than the conventional stick stuff after each other approach. I my opnion in this way it should be easier to keep oversight.

	seg.B0.add_pulse([[10,5]
					 ,[20,5]])
	# B0 is the barrier 0 channel
	seg.B0.add_pulse([[20,0],[30,5], [30,0]])
	seg.B0.add_block(40,70,2)
	seg.B0.add_pulse([[70,0],
					 [80,0],
					 [150,5],
					 [150,0]])

One pulses are added, you can define like:

	SEQ = [['INIT', 1, 0], ['Manip', 1, 0], ['Readout', 1, 0] ]


Which can next be added to the pulse object and next be uploaded.

	p.add_sequence('mysequence', SEQ)

	p.start_sequence('mysequence')

That's it for the moment.
Note that thresolds are chosen automatically. Memory menagment is also taken care of for you  :)

I wrote this text quit late, so don't judge.