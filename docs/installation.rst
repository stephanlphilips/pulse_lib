.. title:: Installation

Installation
============

Pulse_lib
---------

Clone the `sources of Pulse_lib <https://github.com/stephanlphilips/pulse_lib>`_ from GitHub.

.. code-block:: console

	pip install ./pulse_lib

or

.. code-block:: console

	python3 setup.py install

Pulse_lib uses the packages qcodes and qcodes_contrib_drivers.


Keysight PXI
------------

To use pulse_lib with Keysight M3202A AWG and M3102A digitizer you have to install:

* Keysight SD1 software
* Keysight FPGA Test Sync Executive
* Git clone `hvi2-script <https://github.com/QuTech-Delft/hvi2_script>`_ branch SD1_3.1
* Git clone `core tools <https://github.com/stephanlphilips/core_tools>`_

The core tools package contains the HVI2 schedules required to trigger the AWGs and digitzers.
It contains the schedules for video mode and single shot measurements.

The core tools package also contains a driver for the M3102A digitizer.


Tektronix AWG5014
-----------------

No additional software is needed to use pulse_lib with AWG5014.

A schedule to use AWG5014 with Spectrum M4i digitizer and Zurich Instruments UHFLI is included in pulse_lib.


Qblox Pulsar QCM and QRM
------------------------

Note: Support for Qblox is still in development on a separate branch of pulse_lib.

To use pulse_lib with Qblox you have to install:

* package qblox_instruments (available on PyPi)
* `Q1Pulse<https://github.com/sldesnoo-Delft/q1pulse>`_


QuTech QuantumSequencer
-----------------------

The QuantumSequencer uses QuTech developed FPGA images for Keysight M3202A and M3102A.
