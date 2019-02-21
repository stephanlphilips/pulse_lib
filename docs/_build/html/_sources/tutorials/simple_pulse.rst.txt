.. _simple_pulse:

Making simple pulses for DC control of qubits.
==============================================

For most experiments, pulses usually consist out of simple ramps and platea's.
In the following there will be a few examples that show how to program these things.
We will consider two experiments,

	- a pulse that can be used for :ref:`elzerman_pulse`
	- :ref:`a pulse to control a ST qubit<singlet_tiplet_qubit>`

Here you will learn how to make the pulse, but also how the channels are working together.
But before checking the examples, it is best to get a grasp of the theory below.



Pulse construction basics
^^^^^^^^^^^^^^^^^^^^^^^^^

.. _mk_segment_container:

Making a segment container
""""""""""""""""""""""""""

Before we can make any pulse, we need to ask the ``pulselib`` object for a segment container object,

.. code-block:: python

	my_new_pulse_container  = pulse.mk_segment()

Where ``my_new_pulse_container`` is of the type ``segment_container``.
From this object we can now access all the channels. For example, if we want to make a block pulse on the plunger gate of dot 1, we can call,

.. code-block:: python

	start = 10 # ns
	stop = 50 # ns
	amp = 100 # mV
	my_new_pulse_container.P1.add_block(start, stop, amp)

So if we want to apply operations on gates, we can access them like:

.. code-block:: python

	segment_container_obj.my_gate_name.operation_you_want_to_do

In this case, we have quite a few gates available (defined in the :ref:`init of pulse_lib<init_lib>`). We can check them,

.. code-block:: python

	>>> print(my_new_pulse_container.channels)

	[ 'B0', 'P1', 'B1', 'P2', 'B2', 
	  'MW_gate_I', 'MW_gate_Q', 'MW_marker', 
	  'vB0', 'vB1', 'vB2', 'vP1', 'vP2',
	  'qubit_1', 'qubit_2' ]


Segment operations
""""""""""""""""""

A segment is the entity where you can make you pulses. On a segment you can apply an unlimited amount of operations.
When the operation changes something to the waveform, it is usually done by adding it up to what is already there. Thus when retaking the example of before, doing 

.. code-block:: python

	my_new_pulse_container.P1.add_block(start, stop, amp)
	my_new_pulse_container.P1.add_block(start, stop, amp)

is equivalent to doing, 

.. code-block:: python

	my_new_pulse_container.P1.add_block(start, stop, amp)

The operations possible at this moment are:
	
	- adding block shaped pulses
	- adding ramp's
	- adding an arbitrary row of times and voltages that define a pulse shape.
	- add sinus shaped data (for modulation options, use a microwave object)
	- counting 
	- feature possibility add numpy data.

Important operators are:
	
	- slice time : make a new start/end time of the pulse
	- reset time : the time of the last element in the segment will become zero. Now you can restart using time 0.
	- wait : just waits. Example usage, call before reset time, to introduce a few ns buffer after a gate.
	- append : append waveform x to waveform y
	- repeat : repeat the current waveform x times.

To get intuition what each of this operators does, it is best to try to execute (and change) some of the examples given in the beginning.


segment container operations
""""""""""""""""""""""""""""

This is the container that 


How do virtual gates exactly work here?
"""""""""""""""""""""""""""""""""""""""

