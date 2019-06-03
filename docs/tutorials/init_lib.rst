.. _init_lib:

Initializing the library
========================

Before making a pulse, the pulse library needs to know some stuff from you. The main items you need to provide are:

   - QCodes objects of the AWG's (only when using Keysight AWG's)
   - Translation of you sample channels names (e.g. barrier gate 1) to the physical location if the set up (e.g. AWG3 channel 4)
   - Virtual gates and :ref:`virtual gate matrix<virt_gates>` if needed.
   - Virtual IQ channels, if you are doing :ref:`IQ modulation<IQ_mod_basics>`
   - :ref:`Channels delay's<chan_delay>` (e.g. if not all you coaxes have the same length)
   - Which channel need a :ref:`DC offset compenstation<dc_offset_comp>` and what the what the allowed values are.

All these properties are contained in the ``pulselib`` object. Below a complete example is worked out to explaining how to do this.
The source code of the example can be found in ``tutorials/init_pulselib.py`` if you would want to execute it yourselves.

.. _example_init_lib:
Example : initialization for a two-qubit setup
----------------------------------------------

For this setup, we have 6 coax cables going to the sample,

   - B0, B1, B2 : gates that are connected to the barrier gates on the sample.
   - P1, P2 : gates that are connected to the plungers on the sample.
   - MW_gate : a screening gate on the sample that can be used for EDSR.
   - MW_marker : a marker to control on/off state of the vector source.

Practically the things that we want to set are:
	1. Which channels of the AWG are corresponding to the gates on the sample.
	2. Virtual gates to move easily around in charge stability diagrams.
	3. Two virtual IQ channels, one per qubit.
	4. A :ref:`channel delay<chan_delay>` for the MW channel, to compensate for the time that the waveform spends in the vector signal source generator.
	5. Voltage compensation boundaries for :ref:`DC offsets<dc_offset_comp>`.

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
	
	pulse.define_channel('B0','AWG1', 1)
	pulse.define_channel('P1','AWG1', 2)
	pulse.define_channel('B1','AWG1', 3)
	pulse.define_channel('P2','AWG1', 4)
	pulse.define_channel('B2','AWG2', 1)
	pulse.define_channel('MW_gate_I','AWG2', 2)
	pulse.define_channel('MW_gate_Q','AWG2', 3)
	# define marker here!
	pulse.define_marker('MW_marker','AWG2', 4)


Note, when not using the Keysight back-end, you can just call ``p.add_awgs('AWG1', None)``. You will have to feed then the library in another uploader (e.g. qtt virtual AWG).

Step 2 : defining the virtual gates
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This is a quite straightforward process, just define the channels for the virtual gate and their corresponding real channels in a dictionary.

.. code-block:: python
	
	# generate a virtual gate set. You can later on change this object if you want to update the virtual gatre matrix / add more gates.
	# note that you can make multiple sets if you like (e.g. you could also define one with detuning while keeping this ones).  
	virtual_gate_set_1 = virtual_gates_constructor(pulse)
	virtual_gate_set_1.add_real_gates('B0', 'B1', 'B2', 'P1', 'P2')
	virtual_gate_set_1.add_virtual_gates('vB0', 'vB1', 'vB2', 'vP1', 'vP2')
	virtual_gate_set_1.add_virtual_gate_matrix(np.eye(5))

In this case we just constructed a 1 on 1 map of the virtual gates of real gates (diagonal virtual gate matrix). 

The matrix can be updated at any time with:

.. code-block:: python
	
	virtual_gate_set_1.add_virtual_gate_matrix(my_matrix)

When the matrix is updated, you will have to regenerate the segment containers that are currently in memory.
An example how to practically work with virtual gates can be found here [TODO].

Step 3 : defining IQ channels
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you are new to IQ modulation, it is recommended to read the introduction on IQ modulation, here [TODO].

When operating a vector source, you usually have to connect (usually) 3/5 coax cables:

   - I channel (and optionally its image, usually named I+ and I-)
   - Q channel (and optionally its image, usually named Q+ and Q-)
   - marker

Most of the time, you will want to make a virtual channel per qubit, as it allows you to keep easy track of the phase of the qubit. An example of this can be found in the mircowave tutorial.

.. code-block:: python

	# make virtual channels for IQ usage (also here, make one one of these object per MW source)
	IQ_chan_set_1 = IQ_channel_constructor(pulse)
	# set right association of the real channels with I/Q output (note you can define as many as you likes, also channels copies are allowed).
	IQ_chan_set_1.add_IQ_chan("MW_gate_I", IQ_comp = "I", image = "+")
	IQ_chan_set_1.add_IQ_chan("MW_gate_Q", IQ_comp = "Q", image = "+")
	IQ_chan_set_1.add_marker("MW_marker", pre_delay = -15, post_delay = 15)

	# set LO frequency of the MW source. 
	# This can be changed troughout the experiments, but only newly created segments will hold the latest value.
	# alternatively you can also enter a qcodes paramter for automated frequency setting.
	IQ_chan_set_1.set_LO(1e9)

	# name virtual channels to be used.
	IQ_chan_set_1.add_virtual_IQ_channel("MW_qubit_1")
	IQ_chan_set_1.add_virtual_IQ_channel("MW_qubit_2")


.. _pulse_lib_chan_delay:

Step 4 : defining channel delays
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
In our case here, we have to compensate for the fact that some signals take a longer time to get to the sample than other ones. More info on how this is practically accomplished, can be found :ref:`here<chan_delay>`.
Practically, example latencies could be the following:
	
	- 20 ns for the barrier and plunger gates to get from the AWG channels into the fridge.
	- 70 ns to get to microwave channel from the IQ output into the fridge. 5 ns for the signal to reach the vector source, then the signal needs 45ns to be mixed with the carrier frequency, next 20 additional ns are needed to go down in the fridge.
	- 5 ns marker delay

Or translated into python code, 

.. code-block:: python

	pulse.add_channel_delay('B0', 20)
	pulse.add_channel_delay('P1', 20)
	pulse.add_channel_delay('B1', 20)
	pulse.add_channel_delay('P2', 20)
	pulse.add_channel_delay('B2', 20)
	pulse.add_channel_delay('MW_gate_I', 70)
	pulse.add_channel_delay('MW_gate_Q', 70)
	pulse.add_channel_delay('MW_marker', 5)

Note, also negative delays are allowed. All units are in ``ns`` by default.

.. _volt_comp_bound:

Step 5 : Voltage compensation boundaries for DC offsets
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

DC offsets can be corrected automatically, by the software (for more information on how this is done, see :ref:`here<dc_offset_comp>` ). To do this, you will need to specify ranges that can be used for correction. If no range is provided, it means you do not want any correction. 

The range indicates the maximal, minimal voltage that can be provided to make sure that the integral of pulse is zero. You can declare the compensation ranges as:

.. code-block:: python

	pulse.add_channel_compenstation_limit('B0', (-1000, 500))
	pulse.add_channel_compenstation_limit('B1', (-1000, 500))
	pulse.add_channel_compenstation_limit('B2', (-1000, 500))
	pulse.add_channel_compenstation_limit('P1', (-1000, 500))
	pulse.add_channel_compenstation_limit('P2', (-1000, 500))

In this case, no limits were set for the markers and the IQ channels, as these signals go in a 50 ohm matched line (no high pass filtering to compensate for).