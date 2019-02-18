.. pulse_lib documentation master file, created by
   sphinx-quickstart on Mon Feb 18 11:04:17 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to pulse lib's documentation
====================================
This is a pulse library designed to provide all the control signals that are needed control spin qubits coherenly.
A lot of attention is given to performance, structure and ease of use.

Features now include:
	- Support for arbitrary pulse/sine wave based sequences (phase coherent atm)
	- Fully multidimensional. Execute any command as a loop in any dimension.
	- Short and clean syntax. No sympy.
	- Native support for virtual gates
	- IQ toolkit and IQ virtual channels -- Full suppport for single sideband modulation (Along with PM/AM/FM)
	- High speed uploader for Keysight PXI systems which supports upload during playback.

.. toctree::
   :caption: Getting started
   :titlesonly:

   struct
   tutorials/init_lib
   tutorials/simple_pulse
   tutorials/MW_pulse
   tutorials/sequence
   tutorials/looping
   tutorials/reset_time_and_slicing
   tutorials/example_PT
   tutorials/example_RB

When using the library in combination with the keysight PXI AWG's, playback of the waveforms is also supported:
	- How does a upload work? What are the different steps?
	- Your first simple upload.
	- Integrating HVI.
	- More advanced upload options, running uploads at high speeds.

An overview of all the functions in the classes can be found at
	- sequence
	- segment containers
	- segment base
	- segment IQ

API documentation for developers
	- Requesting data from the sequence object.




Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
