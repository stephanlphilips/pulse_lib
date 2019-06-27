# Introduction

This is a pulse library that is build to make pulses that are commonly used to control spin qubits coherenly. A lot of attention is given to performance, structure and ease of use. At the moment the library only has a back-end that is suited for Keysight PXI AWG systems

Features now include:
* support for arbitrary baseband/microwave based sequences
* Fully multidimensional. Execute any command as a loop in any dimension.
* Short and clean syntax. No sympy.
* Native support for virtual gates.
* IQ toolkit and IQ virtual channels -- Full suppport for single sideband modulation (Along with PM/AM/FM) and out of the box pulse shaping. Automatic softare downconversion from MW pulse to IQ, which can later on be upconverted in hardware by mixing with the IF.
* Automatic compenstation for DC offsets.
* High speed uploader for Keysight PXI systems which supports upload during playback (up to ~ 100 experiments per second (record~350))

!! keysight AWG's their waveforms need to have a length of modulo 10 !! (related to the clock of the AWG)
--> segments are concatenated for this purose when uploading (e.g. upload happens in one big chunk)

# Requirements
You need python3.x and a c/c++ compiler. For the c-compiler, the following is recommended
* windows: the Visual Studio SDK C/C++ compiler (tested)
* linux: gcc is fine.
* ox x : gcc or clang both work

To install the upload libraries for the keysight system, you will need:
* the Keysight SD1 software
* openMP (comes by default in visual studio)
(At the moment this is a requirement, will be removed as a requirement at a later time)

# Quick start
The pulse library can be installed by cloning the library from github on your computer.
Navigate in the github folder and run the following in the terminal:
```bash
	python3 setup.py install
```
The python scrip will also take care of compiling the c code. On windows, it is recommended to do this in a Anaconda promt. You will need to run the promt with administarator privelages.

# Documentation
Documentation for the library can be found at:

https://pulse-lib.readthedocs.io

# TODO
TODO list:
* Support for calibarion arguments? -- this should be engineered well.
* HVI2 integration
* add DSP module (must also be C++)

TODO bugs and small things to fix,
* remove finish init
* memory bug : memory is not released automatically after the sequence is done. This should be automated.This can lead to accumulating memory when doing playback and crash the system. Overhead reduced by fixing issue that led to copying things that could be rereferenced, but issue still needs attention. 
* Keysight FPGA pulse correction not there? --> investigation needed. + correction for long time scale pulses
New functionality (prioritized):
* TODO : automate phase compensation of microwave pulses. (master clock already there for the HVI markers)
