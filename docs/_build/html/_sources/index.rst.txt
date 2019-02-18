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

To get started, it is recomended to read through the following documents:

.. toctree::
   :maxdepth: 2

   intro
 



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
