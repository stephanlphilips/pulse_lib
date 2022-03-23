# Changelog
All notable changes to Pulselib will be documented in this file.

## \[1.3.5] - 2022-03-@@
### Added
- Added argument reset_time to wait()

## \[1.3.4] - 2022-03-23
### Improved
- QuantumSequencer use waveform with low sample rate for long DC compensation pulse.

## \[1.3.3] - 2022-03-14
### Fixed
- QuantumSequencer bugs in 1.3.2

## \[1.3.2] - 2022-03-14
### Fixed
- Fixed long wait (> 160 ms) for QuantumSequencer

### Added
- Added attribute 'values' to sequence loop parameters

## \[1.3.1] - 2022-03-07
### Fixed
- Fixed rendering of segments with different sample rates
- Rendering of conditional segments with looping

## \[1.3.0] - 2022-02-22
### Added
- IQ correction of phase, amplitude and offset:
  add_channel_offset, set_qubit_correction_phase, set_qubit_correction_gain
- New interface from hardware class to pulselib: set_channel_attenuations and add_virtual_matrix
- Virtual matrix on top of virtual gates
- Added hw schedule for UHFLI with Tektronix
- Improved release_awg_memory (for Keysight AWG)

### Removed
- pulse.add_channel_compenstation_limit. Correct method is add_channel_compensation_limit

### Fixed
- loops with t_measurement in HVI variable.
- release_awg_memory() for Keysight

## \[1.2.0] - 2021-11-11
First labeled release. Start of dev branch and change logging.
