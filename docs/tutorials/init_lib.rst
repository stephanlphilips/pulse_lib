Tutorial : Initializing the library
===================================

Before making a pulse, the pulse library needs to know some stuff from you. The main items you need to provide are:

   - QCodes objects of the AWG's (only when using Keysight AWG's)
   - Translation of you sample channels names (e.g. barrier gate 1) to the physical location if the set up (e.g. AWG3 channel 4)
   - Virtual gates and virtual gate matrix if needed <reference needed here>
   - Virtual IQ channels, if you are doing IQ modulation <link explaining IQ modulation >
   - Channels delay's (e.g. if not all you coaxes have the same length)

All these properties are contained in the ``pulselib`` object. Below a complete example is worked out to explaining how to do this.
The source code of the example can be found in ``tutorials/init_pulselib.py`` if you would want to execute it yourselves [TODO].

Tutorial example : initialization for a two-qubit setup
-------------------------------------------------------

For this setup, we have 6 coax cables going to the sample,

   - B0, B1, B2 : gates that are connected to the barrier gates on the sample.
   - P1, P2 : gates that are connected to the plungers on the sample.
   - MW_gate : a screening gate on the sample that can be used for EDSR.
   - MW_marker : a marker to control on/off state of the vector source.

Practically the things that we want to set are:
	1. Which channels of the AWG are corresponding to the gates on the sample.
	2. Virtual gates to move easily around in charge stability diagrams.
	3. Two virtual IQ channels, one per qubit.
	4. A channel delay for the MW channel, to compensate for the time that the waveform spends in the vector signal source generator.
	5. Voltage compensation boundaries for DC offsets ``write DC offset article``

Step 1 : initializing the pulse lib object and defining real gates
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The first thing we need to do is to make the ``pulselib`` object, once it is make, we can start to define extra properties.

.. code-block:: python

	from pulse_lib.base_pulse import pulselib
	pulse = pulselib()

That was it, let's now add some AWG's and define the channel names. In this example we will need 8 channels, thus two, 4 channel AWG's.

.. code-block:: python

	# init AWG
	awg1 = keysight_awg.SD_AWG('awg1', chassis = 0, slot= 2, channels = 4, triggers= 8)
	awg2 = keysight_awg.SD_AWG('awg2', chassis = 0, slot= 3, channels = 4, triggers= 8)
	
	# add to pulse_lib
	pulse.add_awgs('AWG1',awg1)
	pulse.add_awgs('AWG2',awg2)

	
	# define channels
	awg_channels_to_physical_locations = dict({'B0':('AWG1', 1), 'P1':('AWG1', 2),
		'B1':('AWG1', 3), 'P2':('AWG1', 4),'B2':('AWG2', 1), 
		'MW_gate_I':('AWG2', 2), 'MW_gate_Q':('AWG2', 3),	
		'MW_marker':('AWG2', 4)})
		
	pulse.define_channels(awg_channels_to_physical_locations)

Note, when not using the Keysight back-end, you can just call ``p.add_awgs('AWG1', None)``. You will have to feed then the library in another uploaded (e.g. qtt virtual AWG).

Step 2 : defining the virtual gates
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This is a quite straightforward process, just define the channels for the virtual gate and their corresponding real channels in a dictionary.

.. code-block:: python

	awg_virtual_gates = {
		'virtual_gates_names_virt' :
			['vB0', 'vB1', 'vB2', 'vP1', 'vP2'],
		'virtual_gates_names_real' :
			['B0', 'B1', 'B2', 'P1', 'P2'],
		'virtual_gate_matrix' : np.eye(5)
	}
	pulse.add_virtual_gates(awg_virtual_gates)
In this case we just constructed a 1 on 1 map of the virtual gates of real gates (diagonal virtual gate matrix). 

The matrix can be updated at any time with:

.. code-block:: python

	pulse.update_virtual_gate_matrix(my_matrix)

When the matrix is updated, it will automatically also update in all segments that have been created before.
An example how to practically work with virtual gates can be found here [TODO].

Step 3 : defining IQ channels
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you are new to IQ modulation, it is recommended to read the introduction on IQ modulation, here [TODO].

When operating a vector source, you usually have to connect (usually) 3 coax cables:

   - I channel
   - Q channel
   - marker
   - ``[opt]`` if you want to go wide-band, you might also need to provide the negative image of the I/Q channel (currently not implemented .., though easy todo).

Most of the time, you will want to make a virtual channel per qubit, as it allows you to keep easy track of the phase of the qubit. An example of this can be found in the mircowave tutorial.

.. code-block:: python

	awg_IQ_channels = {
			'vIQ_channels' : ['qubit_1','qubit_2'],
			'rIQ_channels' : [['MW_gate_I','MW_gate_Q'],['MW_gate_I','MW_gate_Q']],
			'LO_freq' :[MW_source.frequency, 1e9]
			# do not put the brackets for the MW source
			# e.g. MW_source.frequency (this should be a qcodes parameter)
			}
	
	pulse.add_IQ_virt_channels(awg_IQ_channels)

At the moment markers are not added automatically, this is something that will be implemented in the next release of this library.

Step 4 : defining channel delays
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
In our case here, we have to compensate for the fact that some signals take a longer time to get to the sample than other ones. More info on how this is practically accomplished, can be found here ``TODO``.
Practically, example latencies could be the following:
	
	- 20 ns for the barrier and plunger gates to get from the AWG channels into the fridge.
	- 70 ns to get to microwave channel from the IQ output into the fridge. 5 ns for the signal to reach the vector source, then the signal needs 45ns to be mixed with the carrier frequency, next 20 additional ns are needed to go down in the fridge.
	- 5 ns marker delay

Or translated into python code, 

.. code-block:: python

	pulse.add_channel_delay({
		'B0': 20, 
		'P1': 20,
		'B1': 20, 
		'P2': 20,
		'B2': 20, 
		'MW_gate_I': 70, 
		'MW_gate_Q': 70,	
		'MW_marker': 5
	})

Note, also negative delays are allowed. All units are in ``ns`` by default.