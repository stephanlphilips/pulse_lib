# Maintenance moved
NOTE: This repository is not maintained anymore. A maintained repository can be found on GitLab TU Delft:

https://gitlab.tudelft.nl/qutech-qdlabs/pulse_lib

# Introduction

Pulse_lib is a library to control multi-channel AWG pulse sequences and digitizer acquisitions
with a simple API using physical units. It is designed to control qubit experiments, especially quantum dot
and spin qubit experiments.

Sequences can contain direct voltage pulses, phase coherent microwave (MW) pulses, digital markers, triggers,
and digitizer acquisitions.
The MW pulses in a sequence are phase coherent to enable construction of sequences of quantum gates.
Pulse_lib uses IQ output pairs to generate the MW pulses with a vector signal generator.

Parameters of the pulses in a sequence can be swept across a range of values. This turns the sequence in a
multi-dimensional measurement.

Pulses in pulse_lib are specified in the physical units in the context of the target device:
* Amplitudes are specified in millivolts
* Time is specified in nanoseconds
* MW pulses are specified for a specific qubit and resonance and drive frequency in Hz
* Channels are identified with a logical name

Pulse_lib translates the specified pulse sequence to output signals of the AWG. It takes care of:
* Phase coherence of pulses per qubit
* Capacitive coupling of plunger and barrier gates of quantum dots using a virtual matrix
* Signal delays due to vector signal generator and cables
* MW up conversion by vector signal generator
* Attenuators between AWG and target device
* DC charging of bias-T, which acts as a high pass filter for AWG signals

Pulses in pulse_lib can be made conditional on a measurement in the same sequence. However, this feature
is currently only supported by the QuTech QuantumSequencer for Keysight PXI.

Pulse_lib supports the following hardware:
* Keysight PXI M3202A AWG and M3201A digitizer
* Tektronix AWG5014 with Spectrum M4i digitizer
* Qblox Pulsar QCM and QRM
* QuTech QuantumSequencer for Keysight PXI

The communication with the AWGs has been optimized to minimize the overhead between measurements.
The compilation of pulse sequences and the communication with the AWGs will be further optimized
with every new release of the software.

# Requirements
You need python 3.7+ and qcodes.

To use pulse_lib with Keysight you also need Keysight SD1 software, Keysight FPGA Test Sync Executive, hvi2-script
and hvi2 schedules.

To use pulse_lib with Qblox you also need [Q1Pulse](https://github.com/sldesnoo-Delft/q1pulse).

# Quick start
The pulse library can be installed by cloning the library from github on your computer.
Navigate in the github folder and run the following in the terminal:
```bash
	python3 setup.py install
```

# Documentation
Documentation for the library can be found at:

https://pulse-lib.readthedocs.io

