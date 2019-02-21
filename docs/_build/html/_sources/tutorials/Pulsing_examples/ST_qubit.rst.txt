.. _singlet_tiplet_qubit:

Controlling a ST qubit
^^^^^^^^^^^^^^^^^^^^^^
This a tutorial about the coherent control of a ST qubit, but compared to the previous section, will be doing things a bit differently,

	a. there will be a separate segment for initialization, manipulation and readout.
	b. the time_reset function will be used, so we do not need to keep track of times.

As for the experiment, assume we have a qubit system where

	- The tunnel coupling is controllable (e.g. 1KHz --> 1GHz)
	- The Zeeman difference between the qubits is ~ 10MHz.

What we want to do is :math:`\frac{\pi}{2}` pulse on the ST qubit read it out, and that is it.
When looking at the charge stability, we will to pulse to the following locations:

.. figure:: /img/ST_qubit_charge_stability.png

Important locations in gates space:

	a. DC operating point. This is where are in the diagram when the AWG's are off. Let's also assume that we are only very weakly tunnel coupled to the reservoirs.
	b. Loading point for the singlet (0,2) state.
	c. a point in the (0,2) diagram.
	d. readout point.
	e. operating point.

Note that this is just an example, you might need to stop on more locations depending on the experiment you want to do. 

The final pulsing scheme for the total experiment looks like:



Now we have established the pulses, let's try to create them!