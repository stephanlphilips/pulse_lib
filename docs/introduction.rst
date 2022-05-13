.. title:: Introduction

Introduction
============

Pulse_lib is a library to control multi-channel AWG pulse sequences and digitizer acquisitions
with a simple API using physical units. It is designed to control qubit experiments, especially quantum dot
and spin qubit experiments.


Pulses are specified on user-defined channels. A channel can be used for either voltage pulses,
microwave (MW) pulses, marker on-off pulses, or digitizer acquisition.


Gates (voltage channels)
------------------------

A gate is used to apply voltage pulses on the target device. The pulses are output to one AWG channel, or in
the case of a virtual gate to multiple AWG channels.
A virtual gate is a combination of channels that can be used to compensate capacitive coupling.
A virtual gate can also be a combination of other virtual gates, e.g. to define a detuning voltage.

Voltage pulses are additive, i.e. pulses which overlap in time are summed.
The following pulses can be added to a sequence on a (virtual) gate:

* block pulses
* ramps
* sinusoidal pulses
* custom pulses of arbitrary shape

The pulse_lib configuration has settings to compensate the voltage pulses for:

* Signal delays due to cables and filters
* Attenuation between AWG and target device
* DC charging of bias-T, which acts as a high pass filter for AWG signals
* Capacitive coupling of plunger and barrier gates of quantum dots using a virtual matrix


Qubit (MW) channels
-------------------

A qubit channel is used to apply phase coherent MW pulses to the target device.
It should be assigned to an IQ output pair of the AWG. Multiple qubit channels can be assigned
to one IQ output pair. (See hardware for limitations.)

Every qubit channel has a reference frequency (or resonance frequency) used track the
signal phase between pulses.

The following pulses can be added to a sequence on a qubit channel:

* MW pulse with optional envelope for amplitude and phase modulation
* phase shift for virual-Z gates or phase correction after another gate
* Frequency chirps

The pulse_lib configuration has settings to compensate the MW pulses for:

* Signal delays due to vector signal generator and cables
* MW up conversion by vector signal generator
* IQ mixer phase and amplitude errors


Marker channels
---------------

A marker channel is a digital I/O or an AWG output channel used for example to trigger
another instrument or to mute/unmute a signal.

A marker channel can be linked to a MW channel to automatically mute/unmute the output of the MW source.
It can be offset in time to unmute the MW source ahead of the MW pulse.


Acquisition channels
--------------------

Acquisition channels are used to add acquisitions to a sequence. The input for the acquisition channel
can be single digitizer input channel, or an input pair representing I and Q inputs.
The input can be demodulated and phase shifted.

The acquisition can specify to return averaged the data or a down-sampled time trace.
A threshold can be set on the averaged value to convert it to a qubit state measurement.


Physical units
--------------

Pulses in pulse_lib are specified in physical units like they should arrive on the target device:

* Amplitudes are specified in millivolts
* Time is specified in nanoseconds
* MW pulses are specified for a specific qubit and resonance and drive frequency in Hz
* Channels are identified with a logical name

Parameter sweeps
----------------

Parameters of the pulses in a sequence can be swept across a range of values. This turns the sequence in a
multi-dimensional measurement.

Conditional pulses
------------------

Pulses can be made conditional on one or more measurements in the same sequence.
The conditional pulses can be used to create classical controlled qubit gates.

This feature is currently only supported by the QuTech QuantumSequencer for Keysight PXI.

Supported hardware
------------------

Pulse_lib supports the following hardware:

* Keysight PXI M3202A AWG and M3201A digitizer
* Tektronix AWG5014 with Spectrum M4i digitizer, AlazarTech ATS or Zurich Instruments UHFLI
* Qblox Pulsar QCM and QRM
* QuTech QuantumSequencer for Keysight PXI

See the hardware specific sections for supported features and limitations.

The communication with the AWGs has been optimized to minimize the overhead between measurements.
The compilation of pulse sequences and the communication with the AWGs will be further optimized
with every new release of the software.

