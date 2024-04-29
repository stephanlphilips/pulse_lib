# Changelog
All notable changes to Pulselib will be documented in this file.

## \[1.7.24] - 2024-04-29

- Fixed RF generation in video mode with Keysight_QS

## \[1.7.23] - 2024-04-18

- Fixed adding (simultaneous) segments with multiple looping variables.

## \[1.7.22] - 2024-04-16

- Fixed releasing jobs from uploader.
- Small correction to DC compensation in Qblox fast scan 2D.

## \[1.7.21] - 2024-04-04

- Added high resolution sine wave to Qblox.

## \[1.7.20] - 2024-03-26

- Added RF generator to Keysight QS.

## \[1.7.19] - 2024-03-26

- Fix define_digitizer_channel_iq: add RF parameters.

## \[1.7.18] - 2024-03-21

- Fix error check on RF frequency sweep

## \[1.7.17] - 2024-03-21

- Added scan_resonator_frequency for fast digitizer/rf_source frequency sweep on Qblox.
- Log warning when sequence run duration > 3 s.

## \[1.7.16] - 2024-03-13

- Use AWG channel attenuation for Keysight QS sequences.

## \[1.7.15] - 2024-03-13

- Added oscillator triggering for QS mode.

## \[1.7.14] - 2024-03-11

- Clear sequencer memory in FPGA when Keysight_QS is configured.

## \[1.7.13] - 2024-03-08

- Performance improvement in MW pulse processing. (30% gain in precompile of randomized benchmarking.)

## \[1.7.12] - 2024-03-07

- Modified Keysight_QS backend to drive qubits without MW source using plunger and barrier gates.

## \[1.7.11] - 2023-12-20

- Added RF parameters frequency, source_amplitude and phase
- Added set_default_hw_schedule for Keysight and Tektronix
- Validate channel names
- Fixed unit of phase setting for Keysight RF demodulation [rad]

## \[1.7.10] - 2023-12-14

- Fixed acquisition to allow simultaneous acquisition by combining segments (QConstruct simultaneous)
- Added update_end(stop) to segment to extend segment till at least `stop`.

## \[1.7.9] - 2023-12-07

- Fixed Qblox feedback for multiple measurements (requires Q1Pulse v0.11.5)

## \[1.7.8] - 2023-12-04

- Added high resolution rendering of ramps on voltage channels (Qblox)
- Added hres rendering of sine waves on voltage channels (Keysight and Tektronix)
- Added checks on IQ channel configuration.
- Added Keysight AWG RF oscillator control
- Allow n_rep = None also for Keysight
- Fixed issue with bad aligned custom_pulse followed by ramp of > 1000 points.
- Added pre-pulses to Qblox fast scan.

## \[1.7.7] - 2023-10-17

- Corrected fix for mixed measurement assignments and raw measurements in measurement processing.

## \[1.7.6] - 2023-10-17

- Fix mixed measurement assignments and raw measurements in measurement processing.

## \[1.7.5] - 2023-10-13

- Fixed looping of keyword arguments. (It was broken by the performance update of v1.7)
- Removed digitizer triggering using add_HVI_marker (Keysight)

## \[1.7.4] - 2023-10-10

- Fixed Qblox hardware/software threshold data check.
- Raise clearer exception when a state threshold is applied on time trace data.
- Set defaults `selectors=False` and `accept_mask=False` for `get_measurement_results` and `get_measurement_param`

## \[1.7.3] - 2023-09-20

- Added sequencer.recompile to quickly update an existing sequence when the virtual gate matrix has changed
- Corrected qblox version check for marker inversion.

## \[1.7.2] - 2023-09-06

- Use Qblox input channel selection if qblox-instruments v0.11 installed
- Compare results of hardware and software thresholding

## \[1.7.1] - 2023-09-04

- Fixed M4i acquisition data shape
- Qblox feedback corrected for input gain
- Qblox feedback rotation convert radians to degrees

## \[1.7.0] - 2023-08-30

- Added conditional segments for Qblox (requires qblox-instruments v0.9+)
- Added M4i digitizer control to pulse-lib
- Added IQ demodulation to M4iControl
- Performance improvement of 10% up to 5 times in compilation.
- Added looping on segment.sample_rate
- Added aggregate_func to aggregate n samples of 1 measurement to a single value with a user defined function.
- Improved accuracy of hres ramps for Keysight (error is now ~10 ps).
- Do not add DC compensation pulse for Qblox if it is too small. This avoids uploads of sequences that do nothing.

## \[1.6.34] - 2023-08-29

- Improved operations on multi-dimensional looping functions

## \[1.6.33] - 2023-08-17

- Fixed IQ output of qblox_fast_scan

## \[1.6.32] - 2023-08-10

- Removed add_HVI_marker and add_HVI_variable from segment channels.
  Huge performance improvement for GST and RB. Overhead O(N^2) -> O(N)

## \[1.6.31] - 2023-07-28

- Added reload_seq=True to Qblox fast scan 1D and 2D for external sweeps modifying pulse-lib settings
- Fixed data returned by acquisition parameter when no shot is accepted.
- Fixed attenuation qubit channel for Qblox
- Fixed baseband qubit channel for Qblox

## \[1.6.30] - 2023-07-20

- Removed digitizer resonator drive amplitude sweep feature for Qblox. It broke functionality.

## \[1.6.29] - 2023-07-19

- Fixed setting of digitizer demodulation frequency and resonator drive amplitude for Qblox.

## \[1.6.28] - 2023-07-13

- Fixed Qblox uploader

## \[1.6.27] - 2023-07-10

- Fixed retries on Keysight exceptions
- Fixed rendering stacked and/or unaligned sine and custom pulses for Qblox
- Render wave of add_sin() with phase starting at start of wave, i.e. independent of start time of pulse.

## \[1.6.26] - 2023-06-30

- Fixed Qblox time offset bug in previous release.

## \[1.6.25] - 2023-06-30

- Fixed merging of phase shift and MW pulse in conditional segment (Keysight_QS)
- Fixed Qblox uploader with unaligned long pulses and gap between pulses.
- Fixed Keysight release_all_awg_memory
- Added measure till end of sequence (time trace feature)

## \[1.6.24] - 2023-06-19

- Fixed Qblox uploader with unaligned pulses.

## \[1.6.23] - 2023-05-27

- Fixed Keysight FPGA marker upload time rounding (Keysight and Keysight_QS).

## \[1.6.22] - 2023-05-23

- Fixed QuantumSequencer (Keysight_QS) long pulses with post-phase shift.

## \[1.6.21] - 2023-05-17

- Fixed Qblox uploader for unaligned MW pulses > 200 ns

## \[1.6.20] - 2023-05-08

- Fixed Keysight digitizer configuration for time-traces
- Added total_time to sequence and metadata

## \[1.6.19] - 2023-05-04

- Fixed Qblox uploader
- Fixed digitizer configuration for time-traces with Keysight uploaders

## \[1.6.18] - 2023-05-03

- Added histograms to measurement parameter

## \[1.6.17] - 2023-05-02

- Render pulses and ramps on Qblox with 1 ns resolution.
- Removed segment_pulse.add_ramp(), because it gave too many errors.

## \[1.6.16] - 2023-04-19
- Renamed idle frequency and reference frequency to qubit resonance frequency.
- Allow overrule of qubit resonance frequency in sequence.
- Added FrequencyUndefined for not-coherent pulsing with NCO frequency 0.0 between pulses.
- Deprecate segment_pulse.add_ramp in favour of add_ramp_ss
- Added configuration of digitizer to Keysight_QS uploader
- Added at(index) to loop_obj to get value at sequence index.
- Corrected phase for MW pulses on Qblox.
- Added py.typed for mypy
- Merge marker pulses if < 10 ns between pulses.

## \[1.6.15] - 2023-03-06

- Fixed rounding errors on segment boundaries in Qblox uploader.
- Qblox uploader assign IQ marker to sequencer of first qubit on IQ output.
- Refactored IQ channel definition: pulse_lib.define_iq_channel()
- Added lp.arange() similar to numpy.arange()

## \[1.6.14] - 2023-03-03

- Fixed missing IQ marker around chirp

## \[1.6.13] - 2023-03-01

- Fixed MeasurementMajority to work with pulse_templates.
- Fixed DC compensation on Qblox.
- Fixed some edge cases.

## \[1.6.12] - 2023-02-23

- QuantumSequencer: split long constant waveforms in start and stop waveform.

## \[1.6.11] - 2023-02-17

- Fixed measurement parameter for acquisition with accept_if not None

## \[1.6.10] - 2023-02-13

- Round acquisition time towards zero in uploader
- Fix for measurement expressions in sequence
- Fix for simultaneous driving
- Fixed majority vote in expression

## \[1.6.9] - 2023-02-06

- Fixed acquire wait=True with n_repeat
- Added checks on t_measure and sample rate in read_channels.
- Fixed actual number of points and interval for acquisition:
  n_samples, interval = uploader.actual_acquisition_points(acquisition_channel, t_measure, sample_rate)
- Replaced logging by logger

## \[1.6.8] - 2023-02-02

- Fixed setpoints of measurement parameter for time traces > 2 seconds.
- Fixed simultaneous pulses with indexing of segments on looping parameter.
- Calculate timeout for get_acquisition_data from sequence.

## \[1.6.7] - 2023-02-01

- Added continuous mode for Keysight AWG.

## \[1.6.6] - 2023-01-30

- Fixed errors when slicing segments after adding dimension with update_dim.
- Remove phase shifts of 0.0 rad before rendering.
- Fixed error in KeysightQS with phase-shift after MW pulse.
- Added tests.utils.last_upload.get_last_upload to analyze upload after exception

## \[1.6.5] - 2023-01-23

- Fixed numpy deprecation of np.int and np.float.

## \[1.6.4] - 2023-01-19

### Fixed
- Qblox uploader overlapping pulses error

## \[1.6.3] - 2023-01-19

### Added
- parameter iq_mode to sequencer.get_measurement_parameter(), get_measurement_data(), Qblox fast scan
- read_channels() to read digitizer channels without pulsing AWG. (Currently Qblox only)
- Keysight AWG mock rendering at 10 GSa/s using digital filter setting.
- acquire(... wait=True) to add wait time for acquisition.

### Fixed
- Timing of Keysight_QS with 2022 firmware
- Keysight_QS rf_source bug
- Fixed n_samples in measurement_param
- Fixed Qblox sequence generation when no pulses on AWG channels

### Removed
- get_acquisition_param. It was deprecated for quite some time.

## \[1.6.2] - 2022-12-22

### Added
- pulselib.get_channel_attenuations()

### Changed
- Sequencer upload and play only retry upload after specific Keysight exception.
- Chirp is stored as an instruction to be rendered by backend.

## \[1.6.1] - 2022-12-06

### Added
- start_times to measurement_description

### Changed
- measurement names do not have to be unique. A postfix makes them unique.

## \[1.6.0] - 2022-11-25

### Attention !!
- Digitizer trigger timing has been corrected for channel delays !!
- Digitizer RF source API has changed.
- Looping has been refactored to add functionality and improve performance.
- Important internal interface changes are marked with [v1.6.0] in the code.

### Added
- Added segment_container.update_dim for looping with index
- Added numpy array operations to looping variables
- Added looping.array to loop over an arbitrary array of values
- Added digitizer_channel.delay
- Added `prolongation_ns` to RF source configuration
- Added `iq_complex` to fast_scan1D_param and fast_scan2D_param

### Changed
- Changed set_digitizer_rf_source `trigger_offset_ns` to `startup_time_ns`.
- Refactored looping and segment indexing:
  - Improved performance of looping arguments
  - Improved performance of segment_container.reset_time()
  - Improved performance of segment and segment_container indexing
  - Improved performance of sequence pre-rendering
  - Reduced memory usage
  - Added sanity checks on looping arguments
- Added checks on frequency when rendering MW pulses

### Removed
- Removed segment_container append(), slice_time(), last_edit
- Removed segment_IQ.add_global_phase()

### Fixed
- Fixed addition of 2 segment_containers to construct simultaneous driving
- Tektronix marker on unused AWG channel
- Measurement result inversion with zero_on_high
- Multiple measurements on 1 channel with down-sampling

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
