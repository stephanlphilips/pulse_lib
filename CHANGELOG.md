# Changelog
All notable changes to Pulselib will be documented in this file.

## \[1.5.0] - 2022-05-@@@
### Added
- Support for Qblox modules: QCM and QRM
- Added sequencer.get_measurement_data
- Added sequencer.set_acquisition
- Added sequencer.get_acquisition_param with automatic upload/play
- Added channel acquire(..., n_repeat= , interval= )
- Added qblox_fast_scan 1D and 2D


## \[1.4.0] - 2022-05-17
### Changed
- Renamed Tektronix backend 'Tektronix_5014' after refactoring:
    - Faster and allow fast switching between multiple uploaded sequences.
    - Amplitude output has been corrected. It is 2x previous output. Correct attenuation per channel!!
    - Use sequence.play(release=False) to call play multiple times after a single upload.
    - Use infinite looping when n_rep > 65535
### Added
- Added sequencer.repetition_aligment to align the duration of the sequence with an external clock or signal frequency.
  (currently only implemented for Tektronix)
- Added argument reset_time to wait()
- Several pages of documentation (not finished yet..)

## \[1.3.5] - 2022-03-29
### Fixed
- Error when rendering section with low sample rate extends into segment with high sample rate.

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
