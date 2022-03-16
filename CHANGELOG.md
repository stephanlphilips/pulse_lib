# Changelog
All notable changes to Pulselib will be documented in this file.

## \[1.4.0] - @@@@
### Added
- Support for Qblox modules: QCM and QRM
- Added sequencer.get_measurement_data
- Added sequencer.set_acquisition
- Added sequencer.get_acquisition_param with automatic upload/play
- Added acquire(..., n_repeat=None, interval=None)
- Added qblox_fast_scan 1D and 2D
- Improved sequencer sweep index parameters

## \[1.3.2] - 2022-03014
## \[1.3.3] - 2022-03-014
### Fixed
- QuantumSequencer bugs in 1.3.2

## \[1.3.2] - 2022-03-014
### Fixed
- Fixed long wait (> 160 ms) for QuantumSequencer

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
