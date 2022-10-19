# Changelog
All notable changes to Pulselib will be documented in this file.

## \[1.6.0] - 2022-10-@@@

- Refactored looping and segment indexing:
  - Added segment_container.update_dim
  - Added numpy array operations to looping variables
  - Improved performance of looping arguments
  - Improved performance of segment_container.reset_time()
  - Improved performance of segment and segment_container indexing
  - Improved performance of sequence pre-rendering
  - Reduced memory usage
  - Added sanity checks on looping arguments
- Removed segment_container append(), slice_time(), last_edit
- Removed segment_IQ.add_global_phase()
- Fixed addition of 2 segment_containers to construct simultaneous driving
- Added checks on frequency when rendering MW pulses
- Internal interface changes marked with [v1.6.0]

## \[1.5.6] - 2022-10-10

### Fixed
- Fixed error in measurement parameter when no shot is accepted

## \[1.5.5] - 2022-09-27

### Fixed
- Incorrect setpoints in measurement_param.add_derived_param.

## \[1.5.4] - 2022-09-23

### Added
- Added endpoint to looping.linspace, logspace and geomspace.
- Added more data selection options to get_measurement_results()
- Added addition of segment_containers to construct simultaneous driving
- Added sequence_builder.add_simultaneous()

### Changed
- Do not add axis 'repetition' if sequencer.n_rep == None.

### Fixed
- Fixed set_acquisition with t_measure and sample_rate
- Fixed get_measurement_param with iq_complex=False
- Fixed high memory usage due to unlimited waveform cache.

## \[1.5.3] - 2022-08-05

### Fixed
- Fixed looping and HVI marker
- Added retries to sequence upload and play

## \[1.5.2] - 2022-08-03

### Fixed
- sequencer.get_channel_data and get_measurement_param when using play() with index
  as done by core-tools.

## \[1.5.1] - 2022-08-03
### Added
- Added sequencer.plot()

### Fixed
- Fixed looping t_off of add_HVI_marker
- Fixed plotting after mk_sequence

## \[1.5.0] - 2022-07-25
### Added
- Support for Qblox modules: QCM and QRM
- Added sequencer.get_channel_data
- Added sequencer.set_acquisition
- Added sequencer.get_measurement_param with automatic upload/play and thresholding
- Added sequencer.get_measurement_results
- Added channel acquire(..., n_repeat= , interval= )
- Added qblox_fast_scan 1D and 2D
- Added digitizer configuring by M3202A_Uploader.

### Changed
- Default channel offset is None to allow configuration outside of pulselib

### Fixed
- Scaling of custom pulses with virtual gates. Custom function now always called with amplitude of original pulse.


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
