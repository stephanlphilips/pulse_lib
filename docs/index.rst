.. pulse_lib documentation master file, created by
   sphinx-quickstart on Mon Feb 18 11:04:17 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Pulse_lib
=========

Pulse_lib is a library to control multi-channel AWG pulse sequences and digitizer acquisitions
with a simple API using physical units. It is designed to control qubit experiments, especially quantum dot
and spin qubit experiments.

Sequences can contain direct voltage pulses, phase coherent microwave (MW) pulses, digital markers, triggers,
and digitizer acquisitions. Parameters of the pulses in a sequence can be swept across a range of values. This turns the sequence in a
multi-dimensional measurement.

Pulse_lib translates the specified pulse sequence to output signals of the AWG. It takes care of:

* Phase coherence of pulses per qubit
* Capacitive coupling of plunger and barrier gates of quantum dots using a virtual matrix
* Signal delays due to vector signal generator and cables
* MW up conversion by vector signal generator
* Attenuators between AWG and target device
* DC charging of bias-T, which acts as a high pass filter for AWG signals

Pulses can be conditional on a measurement in the same sequence. However, this feature
is currently only supported by the QuTech QuantumSequencer for Keysight PXI.

Pulse_lib supports the following hardware:

* Keysight PXI M3202A AWG and M3201A digitizer
* Tektronix AWG5014 with Spectrum M4i digitizer
* Qblox Pulsar QCM and QRM
* QuTech QuantumSequencer for Keysight PXI


.. toctree::
	:maxdepth: 2
	:caption: Getting started
	:name: getting_started

	introduction
	installation
	tutorials/basic_example

.. toctree::
	:maxdepth: 2
	:caption: Signal physics

	signals/delays
	signals/attenuation
	signals/bias_T
	signals/cross_capacitance
	signals/iq_modulation

.. toctree::
	:maxdepth: 1
	:caption: User Guide

	user/configuration
	user/segments
	user/timeline
	user/voltage_channels
	user/mw_channels
	user/marker_channels
	user/digitizer_channels
	user/parameter_sweep
	user/plotting_segments
	user/acquisition_parameter
	user/executing_sequence

*TODO:*


.. Getting started:

..	- :ref:`struct_lib`
..	- :ref:`init_lib`
..	- :ref:`simple_pulse`


..    struct
..    tutorials/init_lib
..    tutorials/simple_pulse
..    tutorials/MW_pulse
..    tutorials/sequence
..    tutorials/looping
..    tutorials/reset_time_and_slicing
..    tutorials/example_PT
..    tutorials/example_RB


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
