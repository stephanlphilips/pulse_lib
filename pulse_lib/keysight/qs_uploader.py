import time
import math
import numpy as np
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from uuid import UUID

from .sequencer_device import SequencerInfo, SequencerDevice
from .qs_conditional import get_conditional_channel, get_acquisition_names, QsConditionalSegment
from .qs_sequence import (
    AcquisitionSequenceBuilder,
    IQSequenceBuilder,
    SequenceConditionalEntry,
    RFSequenceBuilder
    )

from pulse_lib.segments.data_classes.data_IQ import IQ_data_single, Chirp
from pulse_lib.segments.conditional_segment import conditional_segment
from pulse_lib.tests.mock_m3202a_qs import AwgInstruction, AwgConditionalInstruction
from pulse_lib.tests.mock_m3102a_qs import DigitizerInstruction
from pulse_lib.segments.utility.rounding import iround
from pulse_lib.uploader.uploader_funcs import (
        get_iq_nco_idle_frequency, merge_markers, get_sample_rate)

logger = logging.getLogger(__name__)


class AwgConfig:
    MAX_AMPLITUDE = 1500  # mV
    ALIGNMENT = 10  # waveform must be multiple 10 bytes


class QsUploader:
    verbose = False
    # Option to disable use of sequencers to test with fully generated waveforms.
    use_iq_sequencers = True
    use_baseband_sequencers = False
    use_digitizer_sequencers = True

    def __init__(self, awg_devices, awg_channels, marker_channels,
                 IQ_channels, qubit_channels, digitizers, digitizer_channels):
        self.AWGs = awg_devices
        self.awg_channels = awg_channels
        self.marker_channels = marker_channels
        self.IQ_channels = IQ_channels
        self.qubit_channels = qubit_channels
        self.digitizers = digitizers
        self.digitizer_channels = digitizer_channels

        self.jobs = []
        self.acq_description = None

        self._add_sequencers(awg_devices, awg_channels, IQ_channels)

        self.release_all_awg_memory()
        self._config_marker_channels()
        self._check_digitizer_channels()
        self._configure_rf_sources()
        self._configure_rf_sequencers()

    @property
    def supports_conditionals(self):
        return True

    def _add_sequencers(self, AWGs, awg_channels, IQ_channels):
        sequencer_devices: Dict[str, SequencerDevice] = {}
        self.sequencer_devices = sequencer_devices
        self.sequencer_channels:Dict[str, SequencerInfo] = {}
        # collect output channels. They should not be rendered to full waveforms
        self.sequencer_out_channels:List[str] = []

        for awg_name, awg in AWGs.items():
            if hasattr(awg, 'get_sequencer'):
                sequencer_devices[awg_name] = SequencerDevice(awg)

        for IQ_channel in IQ_channels.values():
            iq_pair = IQ_channel.IQ_out_channels
            if len(iq_pair) > 2:
                raise Exception(f'IQ-channel should have 2 awg channels '
                                f'({iq_pair})')
            iq_awg_channels = [awg_channels[iq_channel_info.awg_channel_name] for iq_channel_info in iq_pair]
            awg_names = [awg_channel.awg_name for awg_channel in iq_awg_channels]
            if not any(awg_name in sequencer_devices for awg_name in awg_names):
                continue

            if len(awg_names) == 2 and awg_names[0] != awg_names[1]:
                raise Exception(f'IQ channels should be on 1 awg: {iq_pair}')

            self.sequencer_out_channels += [awg_channel.name for awg_channel in iq_awg_channels]

            seq_device = sequencer_devices[awg_names[0]]
            channel_numbers = [awg_channel.channel_number for awg_channel in iq_awg_channels]
            if len(channel_numbers) == 2:
                sequencers = seq_device.add_iq_channel(IQ_channel, channel_numbers)
            elif len(channel_numbers) == 1:
                sequencers = seq_device.add_drive_channel(IQ_channel, channel_numbers[0])
            else:
                raise Exception(
                    f"Unsupported IQ configuration with channel numbers {channel_numbers} for {IQ_channel.name}")
            self.sequencer_channels.update(sequencers)

        # # @@@ commented out, because base band channels do not yet work.
        # for awg_channel in awg_channels.values():
        #     if (awg_channel.awg_name in obj.sequencer_devices
        #         and awg_channel.name not in obj.sequencer_out_channels):
        #         seq_device = obj.sequencer_devices[awg_channel.awg_name]
        #         bb_seq = seq_device.add_bb_channel(awg_channel.channel_number, awg_channel.name)

        #         obj.sequencer_channels[bb_seq.channel_name] = bb_seq
        #         obj.sequencer_out_channels += [bb_seq.channel_name]

        # for dev in obj.sequencer_devices.values():
        #     awg = dev.awg
        #     for i,seq_info in enumerate(dev.sequencers):
        #         if seq_info is None:
        #             continue
        #         seq = awg.get_sequencer(i+1)
        #         if seq_info.frequency is None:
        #             # sequencer output is A for channels 1 and 3 and B for 2 and 4
        #             output = 'BA'[seq_info.channel_numbers[0] % 2]
        #             seq.set_baseband(output)
        #         else:
        #             seq.configure_oscillators(seq_info.frequency, seq_info.phases[0], seq_info.phases[1])

    def _check_digitizer_channels(self):
        for name, channel in self.digitizer_channels.items():
            dig_name = channel.module_name
            dig = self.digitizers[dig_name]
            in_ch = channel.channel_numbers
            # Note: Digitizer FPGA returns IQ in complex value.
            #       2 input channels are used with external IQ demodulation without processing in FPGA.
            acq_mode = dig.get_channel_acquisition_mode(in_ch[0])
            if len(in_ch) == 2 and acq_mode in [2, 3, 4, 5]:
                print(f"Warning channel '{name}' with acquisition mode {acq_mode}' should be configured "
                      "in pulse-lib with 1 input channel. Use pulselib.define_digitizer_channel()")
            elif len(in_ch) == 1 and channel.iq_out and acq_mode in [0, 1]:
                print(f"Warning channel '{name}' with acquisition mode {acq_mode}' only has real-valued data, "
                      "but `iq_out` is set to True. This will have no effect.")
            if acq_mode in [2, 3] and channel.frequency is not None:
                if channel.hw_input_channel is None:
                    print(f"Warning input for channel '{name}' not specified, using {in_ch[0]} by default. "
                          "Configure the input channel with pulselib.set_digitizer_hw_input_channel()")

    def _config_marker_channels(self):
        for channel in self.marker_channels.values():
            awg_name = channel.module_name
            awg = self.AWGs[awg_name]
            if channel.channel_number == 0:
                awg.configure_marker_output(invert=channel.invert)
            else:
                offset = 0
                amplitude = channel.amplitude/1000
                if channel.invert:
                    offset = amplitude
                    amplitude = -amplitude

                awg.set_channel_amplitude(amplitude, channel.channel_number)
                awg.set_channel_offset(offset, channel.channel_number)

    def _configure_rf_sources(self):
        # NOTE: only works for M3202A_fpga driver.
        awg_oscillators = None
        for dig_ch in self.digitizer_channels.values():
            if dig_ch.rf_source is not None and dig_ch.frequency is not None:
                rf_source = dig_ch.rf_source
                awg_name, awg_ch = rf_source.output
                awg = self.AWGs[awg_name]
                if not hasattr(awg, 'set_lo_mode'):
                    if hasattr(awg, 'get_sequencer'):
                        # Add oscillator in _configure_lo_sequencers
                        continue
                    else:
                        raise Exception('RF generator must be configured on module with M3202A_fpga driver '
                                        'or M3202A_QS driver')

                if awg_oscillators is None:
                    awg_oscillators = AwgOscillators(delay=rf_source.delay,
                                                     startup_time=rf_source.startup_time_ns,
                                                     prolongation_time=rf_source.prolongation_ns,
                                                     mode=rf_source.mode)
                else:
                    if (rf_source.delay != awg_oscillators.delay
                            or rf_source.startup_time_ns != awg_oscillators.startup_time
                            or rf_source.prolongation_ns != awg_oscillators.prolongation_time
                            or rf_source.mode != awg_oscillators.mode):
                        raise Exception('RF source delay, startup time, prolongation time and mode '
                                        'must be equal for all oscillators')

                osc_num = 0
                for osc in awg_oscillators.oscillators:
                    if osc[:2] == (awg_name, awg_ch):
                        osc_num += 1
                        if osc_num >= 4:
                            raise Exception(f'Too many RF oscillators on {awg_name} channnel {awg_ch}')
                osc = (awg_name, awg_ch, osc_num)
                awg_oscillators.oscillators.append(osc)
                awg_oscillators.dig2osc[dig_ch.name] = osc
                awg.set_lo_mode(awg_ch, True)
                amplitude = rf_source.amplitude / rf_source.attenuation
                enable = rf_source.mode == 'continuous'
                awg.config_lo(awg_ch, osc_num, enable, dig_ch.frequency, amplitude)

        self._awg_oscillators = awg_oscillators

    def _configure_rf_sequencers(self):
        rf_sequencers: Dict[str, SequencerInfo] = {}
        self.rf_sequencers = rf_sequencers
        for dig_ch in self.digitizer_channels.values():
            if dig_ch.rf_source is not None and dig_ch.frequency is not None:
                rf_source = dig_ch.rf_source
                awg_name, awg_ch = rf_source.output
                awg = self.AWGs[awg_name]
                if not hasattr(awg, 'get_sequencer'):
                    continue

                seq_device = self.sequencer_devices[awg_name]
                name = dig_ch.name + '_RF'
                rf_sequencers[name] = seq_device.add_nco_channel(name, awg_ch)

    def get_effective_sample_rate(self, sample_rate):
        """
        Returns the sample rate that will be used by the Keysight AWG.
        This is the a rate >= requested sample rate.
        """
        awg = list(self.AWGs.values())[0]
        return awg.convert_prescaler_to_sample_rate(awg.convert_sample_rate_to_prescaler(sample_rate))

    def actual_acquisition_points(self, acquisition_channel, t_measure, sample_rate):
        '''
        Returns the actual number of points and interval of an acquisition.
        '''
        dig_ch = self.digitizer_channels[acquisition_channel]
        digitizer = self.digitizers[dig_ch.module_name]
        if hasattr(digitizer, 'actual_acquisition_points'):
            # number of points should be equal for all channels. Request for 1 channel.
            channel_number = dig_ch.channel_numbers[0]
            n_samples, interval = digitizer.actual_acquisition_points(channel_number, t_measure, sample_rate)
        else:
            # use old function and assume the digitizer is NOT in MODES.NORMAL
            n_samples = digitizer.get_samples_per_measurement(t_measure, sample_rate)
            interval = int(max(1, round(100e6 / sample_rate))) * 10
        return n_samples, interval

    def get_roundtrip_latency(self):
        # TODO @@@ put in configuration file.
        # awg FPGA processing latency
        awg_latency = 75
        # awg fpga ch out -> dig fpga ch in
        awg2dig = 310
        # dig FPGA processing latency
        dig_latency = 210
        margin = 10

        return awg_latency + awg2dig + dig_latency + margin

    def create_job(self, sequence, index, seq_id, n_rep, sample_rate, neutralize=True, alignment=None):
        # TODO @@@ implement alignment
        # remove any old job with same sequencer and index
        self.release_memory(seq_id, index)
        return Job(self.jobs, sequence, index, seq_id, n_rep, sample_rate, neutralize)

    def add_upload_job(self, job):
        '''
        add a job to the uploader.
        Args:
            job (upload_job):
                upload_job object that defines what needs to be uploaded and possible post processing of
                the waveforms (if needed)
        '''
        '''
        Class taking care of putting the waveform on the right AWG.

        Steps:
        1) get all the upload data
        2) perform DC correction (if needed)
        3) convert data in an aprropriate upload format
        4) start upload of all data
        5) store reference to uploaded waveform in job
        '''
        start = time.perf_counter()

        self.jobs.append(job)  # @@@ add loaded=True to job.

        aggregator = UploadAggregator(self.AWGs, self.digitizers, self.awg_channels,
                                      self.marker_channels, self.digitizer_channels,
                                      self.qubit_channels, self.sequencer_channels,
                                      self.sequencer_out_channels, self.rf_sequencers)

        aggregator.upload_job(job, self.__upload_to_awg)

        duration = time.perf_counter() - start
        logger.info(f'generated upload data ({duration*1000:6.3f} ms)')

    def __upload_to_awg(self, channel_name, waveform):
        # vmin = waveform.min()
        # vmax = waveform.max()
        # length = len(waveform)
        # logger.debug(f'{channel_name}: V({vmin*1000:6.3f}, {vmax*1000:6.3f}) {length}')
        if channel_name in self.awg_channels:
            awg_name = self.awg_channels[channel_name].awg_name
        elif channel_name in self.marker_channels:
            awg_name = self.marker_channels[channel_name].module_name
        else:
            raise Exception(f'Channel {channel_name} not found in configuration')
        awg = self.AWGs[awg_name]
        wave_ref = awg.upload_waveform(waveform)
        return wave_ref

    def __upload_markers(self, channel_name, table):
        start = time.perf_counter()
        if channel_name not in self.marker_channels:
            raise Exception(f'Channel {channel_name} not found in configuration')
        marker_channel = self.marker_channels[channel_name]
        awg_name = marker_channel.module_name
        awg = self.AWGs[awg_name]
        awg.load_marker_table(table)
        if QsUploader.verbose:
            logger.debug(f'marker for {channel_name} loaded in {(time.perf_counter()-start)*1000:4.2f} ms')

    def __get_job(self, seq_id, index):
        """
        get job data of an uploaded segment
        Args:
            seq_id (uuid) : id of the sequence
            index (tuple) : index that has to be played
        Return:
            job (upload_job) :job, with locations of the sequences to be uploaded.
        """
        for job in self.jobs:
            if job.seq_id == seq_id and job.index == index and not job.released:
                return job

        logger.error(f'Job not found for index {index} of seq {seq_id}')
        raise ValueError(f'Sequence with id {seq_id}, index {index} not found')

    def _configure_digitizers(self, job):
        if not job.acquisition_conf.configure_digitizer:
            return
        '''
        Configure per digitizer channel:
            n_triggers: job.n_triggers per channel
            t_measure: job.t_measure per channel
            downsampled_rate:
            acquisition_mode: set externally
            scale, impedance: set externally
        '''
        n_rep = job.n_rep if job.n_rep else 1
        total_seconds = job.playback_time * n_rep * 1e-9
        timeout = int(total_seconds*1.1) + 3

        enabled_channels = defaultdict(list)
        channels = job.acquisition_conf.channels
        if channels is None:
            channels = list(job.n_acq_samples.keys())
        if QsUploader.use_digitizer_sequencers:
            # sample_rate is in digitizer instruction table
            sample_rate = None
        else:
            sample_rate = job.acquisition_conf.sample_rate
        if job.acquisition_conf.f_sweep is not None:
            raise Exception("In sequence resonator frequency sweep not supported for Keysight")

        for channel_name, t_measure in job.t_measure.items():
            if channel_name not in channels:
                continue
            n_triggers = job.n_acq_samples[channel_name]
            channel_conf = self.digitizer_channels[channel_name]
            dig_name = channel_conf.module_name
            dig = self.digitizers[dig_name]
            for ch in channel_conf.channel_numbers:
                n_rep = job.n_rep if job.n_rep else 1
                dig.set_daq_settings(ch, n_triggers*n_rep, t_measure,
                                     downsampled_rate=sample_rate)
                enabled_channels[dig_name].append(ch)

        # disable not used channels of digitizer
        for dig_name, channel_nums in enabled_channels.items():
            dig = self.digitizers[dig_name]
            dig.set_operating_mode(2)  # HVI
            dig.set_data_handling_mode(0)  # Full mode, no averaging of time or repetitions.
            dig.set_active_channels(channel_nums)
            if hasattr(dig, 'set_timeout'):
                dig.set_timeout(timeout)

        self.acq_description = AcqDescription(job.seq_id, job.index, channels,
                                              job.n_acq_samples, enabled_channels,
                                              job.n_rep,
                                              job.acquisition_conf.average_repetitions)

    def _configure_rf_oscillators(self, job):
        for ch_name, channel_conf in self.digitizer_channels.items():
            dig_name = channel_conf.module_name
            dig = self.digitizers[dig_name]
            acq_mode = dig.get_channel_acquisition_mode(channel_conf.channel_numbers[0])
            if acq_mode in [4, 5]:
                # Set phase for IQ demodulated input
                if channel_conf.phase is not None:
                    for ch in channel_conf.channel_numbers:
                        dig.set_lo(ch, 0, np.degrees(channel_conf.phase))
            if acq_mode in [2, 3]:
                # Set frequency, amplitude and phase for IQ demodulation in FPGA.
                if channel_conf.frequency is not None:
                    for ch in channel_conf.channel_numbers:
                        dig.set_lo(ch,
                                   channel_conf.frequency,
                                   np.degrees(channel_conf.phase),
                                   channel_conf.hw_input_channel,
                                   )
                    if channel_conf.rf_source is not None and self._awg_oscillators is not None:
                        rf_source = channel_conf.rf_source
                        try:
                            osc = self._awg_oscillators.dig2osc[ch_name]
                        except KeyError:
                            logger.error(f"RF for channel {ch_name} not on M3202A_fpga module")
                            continue
                        awg_name, awg_ch, osc_num = osc
                        awg = self.AWGs[awg_name]
                        amplitude = rf_source.amplitude / rf_source.attenuation
                        enable = rf_source.mode == 'continuous'
                        awg.config_lo(awg_ch, osc_num, enable, channel_conf.frequency, amplitude)

    def _get_hvi_params(self, job):
        hvi_params = job.schedule_params.copy()
        if not QsUploader.use_digitizer_sequencers:
            hvi_params.update(
                    {f'dig_trigger_{i+1}': t
                     for i, t in enumerate(job.digitizer_triggers.keys())
                     })
            dig_trigger_channels = {
                    dig_name: [[] for _ in job.digitizer_triggers]
                    for dig_name in self.digitizers.keys()}
            for i, ch_names in enumerate(job.digitizer_triggers.values()):
                for ch_name in ch_names:
                    dig_ch = self.digitizer_channels[ch_name]
                    dig_trigger_channels[dig_ch.module_name][i] += dig_ch.channel_numbers
            hvi_params.update(
                    {f'dig_trigger_channels_{dig_name}': triggers
                     for dig_name, triggers in dig_trigger_channels.items()
                     })

        for awg_name, awg in self.AWGs.items():
            hvi_params[f'use_awg_sequencers_{awg_name}'] = (
                (QsUploader.use_iq_sequencers or QsUploader.use_baseband_sequencers)
                and hasattr(awg, 'get_sequencer'))

        for dig_name, dig in self.digitizers.items():
            hvi_params[f'use_digitizer_sequencers_{dig_name}'] = (
                QsUploader.use_digitizer_sequencers and hasattr(dig, 'get_sequencer'))

        if self._awg_oscillators is not None and self._awg_oscillators.mode == 'pulsed':
            awg_osc = self._awg_oscillators
            t_measure = None
            for dig_ch_name, t_measure_ch in job.t_measure.items():
                if dig_ch_name in awg_osc.dig2osc:
                    if t_measure is None:
                        t_measure = t_measure_ch
                    elif t_measure != t_measure_ch:
                        raise Exception('t_measure must be equal for all RF oscillators')
            if t_measure is not None:
                enabled_los = []
                osc_start_offset = awg_osc.delay - awg_osc.startup_time
                osc_end_offset = awg_osc.delay + awg_osc.prolongation_time + t_measure
                i = 0
                for t, ch_names in job.digitizer_triggers.items():
                    merge = i > 0 and t + osc_start_offset < hvi_params[f'awg_los_off_{i}'] + 50
                    if merge:
                        # merge
                        i -= 1
                    else:
                        hvi_params[f'awg_los_on_{i+1}'] = t + osc_start_offset
                    hvi_params[f'awg_los_off_{i+1}'] = t + osc_end_offset
                    triggered_los = []
                    for ch_name in ch_names:
                        try:
                            osc = awg_osc.dig2osc[ch_name]
                            triggered_los.append(osc)
                        except KeyError:
                            pass
                    if merge:
                        # merge lists
                        enabled_los[-1] = list(set(enabled_los[-1]).union(triggered_los))
                    else:
                        enabled_los.append(triggered_los)
                    i += 1
                hvi_params['switch_los'] = True
                hvi_params['n_switch_los'] = i
                hvi_params['enabled_los'] = enabled_los
            if 'video_mode_channels' in hvi_params:
                video_mode_los = set()
                for dig_name, channels in hvi_params['video_mode_channels'].items():
                    for dig_ch_name, osc in awg_osc.dig2osc.items():
                        dig_channel = self.digitizer_channels[dig_ch_name]
                        if (dig_channel.module_name == dig_name
                                and not set(dig_channel.channel_numbers).isdisjoint(channels)):
                            video_mode_los.add(osc)
                hvi_params['video_mode_los'] = list(video_mode_los)

        return hvi_params

    def play(self, seq_id, index, release_job=True):
        """
        start playback of a sequence that has been uploaded.
        Args:
            seq_id (uuid) : id of the sequence
            index (tuple) : index that has to be played
            release_job (bool) : release memory on AWG after done.
        """
        job = self.__get_job(seq_id, index)
        continuous_mode = getattr(job.hw_schedule, 'script_name', '') == 'Continuous'
        if continuous_mode:
            for awg in self.AWGs.values():
                awg.awg_stop_multiple(0b1111)
        self.wait_until_AWG_idle()

        for channel_name, marker_table in job.marker_tables.items():
            self.__upload_markers(channel_name, marker_table)

        for awg_channel in self.awg_channels.values():
            awg_name = awg_channel.awg_name
            channel_number = awg_channel.channel_number
            # empty AWG queue
            self.AWGs[awg_name].awg_flush(channel_number)
        for marker_channel in self.marker_channels.values():
            awg_name = marker_channel.module_name
            channel_number = marker_channel.channel_number
            if channel_number > 0:
                # empty AWG queue
                self.AWGs[awg_name].awg_flush(channel_number)

        # queue waveforms
        for channel_name, queue in job.channel_queues.items():
            try:
                offset = 0

                if channel_name in self.awg_channels:
                    channel = self.awg_channels[channel_name]
                    awg_name = channel.awg_name
                    channel_number = channel.channel_number
                    amplitude = channel.amplitude if channel.amplitude is not None else AwgConfig.MAX_AMPLITUDE
                    offset = channel.offset if channel.offset is not None else 0
                elif channel_name in self.marker_channels:
                    channel = self.marker_channels[channel_name]
                    awg_name = channel.module_name
                    channel_number = channel.channel_number
                    amplitude = channel.amplitude
                    if channel.invert:
                        offset = amplitude
                        amplitude = -amplitude
                else:
                    raise Exception(f'Undefined channel {channel_name}')

                awg = self.AWGs[awg_name]
                awg.set_channel_amplitude(amplitude/1000, channel_number)
                awg.set_channel_offset(offset/1000, channel_number)

                start_delay = 0  # no start delay
                trigger_mode = 1  # software/HVI trigger
                # cycles = 0 means infinite number of cycles
                cycles = 1 if not continuous_mode else 0
                for queue_item in queue:
                    prescaler = awg.convert_sample_rate_to_prescaler(queue_item.sample_rate)
                    awg.awg_queue_waveform(
                            channel_number, queue_item.wave_reference,
                            trigger_mode, start_delay, cycles, prescaler)
                    trigger_mode = 0  # Auto tigger -- next waveform will play automatically.
            except Exception as ex:
                raise Exception(f'Play failed on channel {channel_name} ({ex})')

        # set offset for IQ channels
        for channel_name, awg_channel in self.awg_channels.items():
            if channel_name not in job.channel_queues:
                awg_name = awg_channel.awg_name
                channel_number = awg_channel.channel_number
                offset = awg_channel.offset if awg_channel.offset is not None else 0
                awg = self.AWGs[awg_name]
                awg.set_channel_offset(offset/1000, channel_number)

        start = time.perf_counter()
        for awg_sequencer in self.sequencer_channels.values():
            if continuous_mode:
                raise Exception('QS sequencers cannot be used in continuous mode. '
                                'Set QsUploader.use_iq_sequencers = False')
            awg = self.AWGs[awg_sequencer.module_name]
            channel_name = awg_sequencer.channel_name
            seq = awg.get_sequencer(awg_sequencer.sequencer_index)
            seq.flush_waveforms()
            schedule = []
            if channel_name in job.iq_sequences:
                t1 = time.perf_counter()
                sequence = job.iq_sequences[awg_sequencer.channel_name]

                # TODO move NCO frequency to iq_sequence object
                qubit_channel = self.qubit_channels[awg_sequencer.channel_name]
                if len(sequence.waveforms) > 0:
                    seq._frequency = get_iq_nco_idle_frequency(job, qubit_channel, index)
                    if seq._frequency is None:
                        raise Exception('Qubit resonance frequency must be configured for QS')
                    if abs(seq._frequency) > 450e6:
                        raise Exception(f'{channel_name} IQ frequency {seq._frequency/1e6:5.1f} MHz is out of range')

                # @@@ IQSequence.upload() OR Sequence.upload()
                for number, wvf in enumerate(sequence.waveforms):
                    seq.upload_waveform(number, wvf.offset, wvf.duration,
                                        wvf.amplitude, wvf.am_envelope,
                                        wvf.frequency, wvf.pm_envelope,
                                        wvf.prephase, wvf.postphase,
                                        wvf.restore_frequency,
                                        append_zero=wvf.restore_frequency)

                t2 = time.perf_counter()
                for i, entry in enumerate(sequence.sequence):
                    if isinstance(entry, SequenceConditionalEntry):
                        schedule.append(AwgConditionalInstruction(i, entry.time_after,
                                                                  wave_numbers=entry.waveform_indices,
                                                                  condition_register=entry.cr))
                    else:
                        schedule.append(AwgInstruction(i, entry.time_after, wave_number=entry.waveform_index))
                t3 = time.perf_counter()
                if QsUploader.verbose:
                    logger.debug(f'{awg_sequencer.channel_name} create waves:{(t2-t1)*1000:6.3f}, '
                                 f'seq:{(t3-t2)*1000:6.3f} ms')
            seq.load_schedule(schedule)

        if QsUploader.verbose:
            logger.debug(f'loaded awg sequences in {(time.perf_counter() - start)*1000:6.3f} ms')

        for rf_sequencer in self.rf_sequencers.values():
            awg = self.AWGs[rf_sequencer.module_name]
            channel_name = rf_sequencer.channel_name
            seq = awg.get_sequencer(rf_sequencer.sequencer_index)
            seq.flush_waveforms()
            schedule = []

            if rf_sequencer.channel_name in job.rf_sequences:
                t1 = time.perf_counter()
                dig_ch = self.digitizer_channels[rf_sequencer.channel_name[:-3]]
                sequence = job.rf_sequences[rf_sequencer.channel_name]
                sequence.set_frequency(dig_ch.frequency)
                if len(sequence.waveforms) > 0:
                    if dig_ch.frequency is None:
                        raise Exception('RF frequency not set for {channel_name}')
                    if abs(dig_ch.frequency) > 450e6:
                        raise Exception(f'{channel_name} RF frequency {dig_ch.frequency/1e6:5.1f} MHz is out of range')

                # oscillator is started with start waveform which has a small offset. Do not set a default frequency.
                seq._frequency = 0
                for number, wvf in enumerate(sequence.waveforms):
                    seq.upload_waveform(number, wvf.offset, wvf.duration,
                                        wvf.amplitude,
                                        am_envelope=1.0,
                                        frequency=wvf.frequency,
                                        prephase=wvf.prephase,
                                        postphase=0,
                                        restore_frequency=False,
                                        append_zero=False)

                t2 = time.perf_counter()
                for i, entry in enumerate(sequence.sequence):
                    schedule.append(AwgInstruction(i, entry.time_after, wave_number=entry.waveform_index))
                t3 = time.perf_counter()
                if QsUploader.verbose:
                    logger.debug(f'{rf_sequencer.channel_name} create waves:{(t2-t1)*1000:6.3f}, '
                                 f'seq:{(t3-t2)*1000:6.3f} ms')
            seq.load_schedule(schedule)

        start = time.perf_counter()
        for dig_channel in self.digitizer_channels.values():
            dig = self.digitizers[dig_channel.module_name]
            if QsUploader.use_digitizer_sequencers and hasattr(dig, 'get_sequencer'):
                seq_numbers = dig_channel.channel_numbers
                for seq_nr in seq_numbers:
                    seq = dig.get_sequencer(seq_nr)
                    sequence = job.digitizer_sequences[dig_channel.name]
                    # @@@ DigSequence.upload()
                    schedule = []
                    for i, entry in enumerate(sequence.sequence):
                        schedule.append(DigitizerInstruction(i, entry.time_after,
                                                             t_measure=entry.t_measure,
                                                             n_cycles=entry.n_cycles,
                                                             measurement_id=entry.measurement_id,
                                                             pxi=entry.pxi_trigger,
                                                             threshold=entry.threshold))
                    seq.load_schedule(schedule)

        if QsUploader.verbose:
            logger.debug(f'loaded dig sequences in {(time.perf_counter() - start)*1000:6.3f} ms')

        self._configure_digitizers(job)
        self._configure_rf_oscillators(job)

        # start hvi (start function loads schedule if not yet loaded)
        schedule_params = self._get_hvi_params(job)
        job.hw_schedule.set_configuration(schedule_params, job.n_waveforms)
        n_rep = job.n_rep if job.n_rep else 1
        run_duration = n_rep * job.playback_time * 1e-9 + 0.1
        if run_duration > 3.0:
            logger.warning(f"Expected duration for point: {run_duration:.1f} s")
        job.hw_schedule.start(job.playback_time, n_rep, schedule_params)

        if release_job:
            job.release()

    def get_channel_data(self, seq_id, index):
        acq_desc = self.acq_description
        if acq_desc.seq_id != seq_id or (index is not None and acq_desc.index != index):
            raise Exception(f'Data for index {index} not available')

        dig_data = {}
        for dig_name in acq_desc.enabled_channels:
            dig = self.digitizers[dig_name]
            dig_data[dig_name] = {}
            active_channels = dig.active_channels
            data = dig.measure.get_data()
            for ch_num, ch_data in zip(active_channels, data):
                dig_data[dig_name][ch_num] = ch_data

        result = {}
        for channel_name in acq_desc.channels:
            channel = self.digitizer_channels[channel_name]
            dig_name = channel.module_name
            in_ch = channel.channel_numbers
            # Note: Digitizer FPGA returns IQ in complex value.
            #       2 input channels are used with external IQ demodulation without processing in FPGA.
            if len(in_ch) == 2:
                raw_I = dig_data[dig_name][in_ch[0]]
                raw_Q = dig_data[dig_name][in_ch[1]]
                if dig.get_channel_acquisition_mode(in_ch[0]) in [2, 3, 4, 5]:
                    # Note: Wrong configuration! len(in_ch) should be 1
                    # phase shift is already applied in HW. Only use data of first channel
                    raw_ch = raw_I
                else:
                    raw_ch = (raw_I + 1j * raw_Q) * np.exp(1j*channel.phase)
            else:
                # this can be complex valued output with LO modulation or phase shift in digitizer (FPGA)
                raw_ch = dig_data[dig_name][in_ch[0]]

            if not channel.iq_out:
                raw_ch = raw_ch.real

            result[channel_name] = raw_ch

        if acq_desc.n_rep:
            for key, value in result.items():
                result[key] = value.reshape((acq_desc.n_rep, -1))
                if acq_desc.average_repetitions:
                    result[key] = np.mean(result[key], axis=0)
        else:
            for key, value in result.items():
                result[key] = value.flatten()

        return result

    def release_memory(self, seq_id=None, index=None):
        """
        Release job memory for `seq_id` and `index`.
        Args:
            seq_id (uuid) : id of the sequence. if None release all
            index (tuple) : index that has to be released; if None release all.
        """
        for job in self.jobs.copy():
            if seq_id is None or (job.seq_id == seq_id and (index is None or job.index == index)):
                job.release()

    def release_all_awg_memory(self):
        for awg in self.AWGs.values():
            for ch in [1, 2, 3, 4]:
                awg.awg_flush(ch)
            if hasattr(awg, 'release_waveform_memory'):
                awg.release_waveform_memory()
            else:
                print('Update M3202A driver')

    def release_jobs(self):
        for job in self.jobs.copy():
            job.release()

    def wait_until_AWG_idle(self):
        '''
        check if the AWG is doing playback, when done, release this function
        '''
        # assume all awg's are used and also all the channels
        channel = list(self.awg_channels.values())[0]
        awg = self.AWGs[channel.awg_name]

        while awg.awg_is_running(channel.channel_number):
            time.sleep(0.001)


@dataclass
class AwgOscillators:
    delay: float
    startup_time: float
    prolongation_time: float
    mode: str
    oscillators: List[Tuple[str, int, int]] = field(default_factory=list)
    dig2osc: Dict[str, Tuple[str, int, int]] = field(default_factory=dict)


@dataclass
class AwgQueueItem:
    wave_reference: object
    sample_rate: float


@dataclass
class AcqDescription:
    seq_id: UUID
    index: List[int]
    channels: List[str]
    n_acq_samples: Dict[str, int]
    enabled_channels: Dict[str, List[int]]
    n_rep: int
    average_repetitions: bool


class Job(object):

    def __init__(self, job_list, sequence, index, seq_id, n_rep, sample_rate, neutralize=True):
        '''
        Args:
            job_list (list): list with all jobs.
            sequence (list of list): list with list of the sequence
            index (tuple) : index that needs to be uploaded
            seq_id (uuid) : if of the sequence
            n_rep (int) : number of repetitions of this sequence.
            sample_rate (float) : sample rate
            neutralize (bool) : place a neutralizing segment at the end of the upload
        '''
        self.job_list = job_list
        self.sequence = sequence
        self.seq_id = seq_id
        self.index = index
        self.n_rep = n_rep
        self.default_sample_rate = sample_rate
        self.neutralize = neutralize
        self.playback_time = 0  # total playtime of the waveform
        self.acquisition_conf = None

        self.released = False

        self.channel_queues = dict()
        self.hw_schedule = None
        logger.debug(f'new job {seq_id}-{index}')

    def add_hw_schedule(self, hw_schedule, schedule_params):
        """
        Add the scheduling to the AWG waveforms.
        args:
            hw_schedule (HardwareSchedule) : schedule for repetitively starting the AWG waveforms
            kwargs : keyword arguments for the hardware schedule (see usage in the examples)
        """
        self.hw_schedule = hw_schedule
        self.schedule_params = schedule_params

    def set_acquisition_conf(self, conf):
        self.acquisition_conf = conf

    def set_feedback(self, condition_measurements):
        # TODO @@@ use preprocessed information for more powerful feedback.
        pass

    def add_waveform(self, channel_name, wave_ref, sample_rate):
        if channel_name not in self.channel_queues:
            self.channel_queues[channel_name] = []

        self.channel_queues[channel_name].append(AwgQueueItem(wave_ref, sample_rate))

    def release(self):
        if self.released:
            logger.warning(f'job {self.seq_id}-{self.index} already released')
            return

        self.upload_info = None
        logger.debug(f'release job {self.seq_id}-{self.index}')
        self.released = True

        for channel_name, queue in self.channel_queues.items():
            for queue_item in queue:
                queue_item.wave_reference.release()

        if self in self.job_list:
            self.job_list.remove(self)

    def __del__(self):
        if not self.released:
            logger.warning(f'Job {self.seq_id}-{self.index} was not released. '
                           'Automatic release in destructor.')
            self.release()


@dataclass
class ChannelInfo:
    # static data
    delay_ns: float = 0
    amplitude: float = 0
    attenuation: float = 1.0
    dc_compensation: bool = False
    dc_compensation_min: float = 0.0
    dc_compensation_max: float = 0.0
    bias_T_RC_time: Optional[float] = None
    # aggregation state
    integral: float = 0.0


@dataclass
class RenderSection:
    sample_rate: float
    t_start: float  # can be negative for negative channel delays
    npt: int = 0

    @property
    def t_end(self):
        return self.t_start + self.npt / self.sample_rate

    def align(self, extend):
        if extend:
            if self.npt < 2000:
                self.npt = 2000
            else:
                self.npt = int((self.npt + AwgConfig.ALIGNMENT - 1) // AwgConfig.ALIGNMENT) * AwgConfig.ALIGNMENT
        else:
            self.npt = int(self.npt // AwgConfig.ALIGNMENT) * AwgConfig.ALIGNMENT


@dataclass
class JobUploadInfo:
    sections: List[RenderSection] = field(default_factory=list)
    dc_compensation_duration: float = 0.0
    dc_compensation_voltages: Dict[str, float] = field(default_factory=dict)


@dataclass
class SegmentRenderInfo:
    # original times from sequence, but rounded for sample rates
    # first segment starts at t_start = 0
    sample_rate: float
    t_start: float
    npt: int
    section: Optional[RenderSection] = None
    offset: int = 0
    # transition when frequency changes: size determined by alignment and channel delays
    n_start_transition: int = 0
    start_section: Optional[RenderSection] = None
    n_end_transition: int = 0
    end_section: Optional[RenderSection] = None

    @property
    def t_end(self):
        return self.t_start + self.npt / self.sample_rate


@dataclass
class RefChannels:
    start_time: float
    start_phase: Dict[str, float] = field(default_factory=dict)
    start_phases_all: List[Dict[str, float]] = field(default_factory=list)


@dataclass
class RfMarkerPulse:
    start: float
    stop: float


class UploadAggregator:
    verbose = False

    def __init__(self, AWGs, digitizers, awg_channels, marker_channels, digitizer_channels,
                 qubit_channels, sequencer_channels, sequencer_out_channels, rf_sequencers):
        self.AWGs = AWGs
        self.digitizers = digitizers
        self.npt = 0
        self.marker_channels = marker_channels
        self.digitizer_channels = digitizer_channels
        self.sequencer_channels = sequencer_channels
        self.sequencer_out_channels = sequencer_out_channels
        self.qubit_channels = qubit_channels
        self.rf_sequencers = rf_sequencers

        self.channels = dict()

        delays = []
        for channel in awg_channels.values():
            info = ChannelInfo()
            self.channels[channel.name] = info

            info.attenuation = channel.attenuation
            info.delay_ns = channel.delay
            info.amplitude = channel.amplitude if channel.amplitude is not None else AwgConfig.MAX_AMPLITUDE
            info.bias_T_RC_time = channel.bias_T_RC_time
            delays.append(channel.delay)

            # Note: Compensation limits are specified before attenuation, i.e. at AWG output level.
            #       Convert compensation limit to device level.
            info.dc_compensation_min = channel.compensation_limits[0] * info.attenuation
            info.dc_compensation_max = channel.compensation_limits[1] * info.attenuation
            info.dc_compensation = info.dc_compensation_min < 0 and info.dc_compensation_max > 0

        for channel in marker_channels.values():
            delays.append(channel.delay - channel.setup_ns)
            delays.append(channel.delay + channel.hold_ns)

        for channel in digitizer_channels.values():
            delays.append(channel.delay)
            if channel.rf_source is not None:
                rf_source = channel.rf_source
                if rf_source.mode == 'pulsed':
                    # HVI2 can only start RF source 30 ns after start of waveform
                    delays.append(rf_source.delay - rf_source.startup_time_ns - 30)
                    delays.append(rf_source.delay + rf_source.prolongation_ns)

        self.max_pre_start_ns = -min(0, *delays)
        self.max_post_end_ns = max(0, *delays)

    def _integrate(self, job):

        if not job.neutralize:
            return

        for iseg, seg in enumerate(job.sequence):
            sample_rate = get_sample_rate(job, seg)

            for channel_name, channel_info in self.channels.items():
                if iseg == 0:
                    channel_info.integral = 0

                if channel_info.dc_compensation:
                    if isinstance(seg, conditional_segment):
                        seg_ch = get_conditional_channel(seg, channel_name)
                    else:
                        seg_ch = seg[channel_name]
                    channel_info.integral += seg_ch.integrate(job.index, sample_rate)
                    if UploadAggregator.verbose:
                        logger.debug(f'Integral seg:{iseg} {channel_name} integral:{channel_info.integral}')

    def _generate_sections(self, job):
        max_pre_start_ns = self.max_pre_start_ns
        max_post_end_ns = self.max_post_end_ns

        self.segments = []
        segments = self.segments
        t_start = 0
        for seg in job.sequence:
            # work with sample rate in GSa/s
            sample_rate = get_sample_rate(job, seg) * 1e-9
            duration = seg.get_total_time(job.index)
            if UploadAggregator.verbose:
                logger.debug(f'Seg duration:{duration:9.3f}')
            npt = iround(duration * sample_rate)
            info = SegmentRenderInfo(sample_rate, t_start, npt)
            segments.append(info)
            t_start = info.t_end

        # sections
        sections = job.upload_info.sections
        t_start = -max_pre_start_ns
        nseg = len(segments)

        section = RenderSection(segments[0].sample_rate, t_start)
        sections.append(section)
        section.npt += iround(max_pre_start_ns * section.sample_rate)

        for iseg, seg in enumerate(segments):
            sample_rate = seg.sample_rate

            if iseg < nseg-1:
                sample_rate_next = segments[iseg+1].sample_rate
            else:
                sample_rate_next = 0

            # create welding region if sample_rate decreases
            if sample_rate < section.sample_rate:
                # welding region is length of padding for alignment + post_stop region
                n_post = iround(((seg.t_start + max_post_end_ns) - section.t_end) * section.sample_rate)
                section.npt += n_post
                section.align(extend=True)

                # number of points of segment to be rendered to previous section
                n_start_transition = iround((section.t_end - seg.t_start)*sample_rate)

                seg.n_start_transition = n_start_transition
                seg.start_section = section

                # start new section
                section = RenderSection(sample_rate, section.t_end)
                sections.append(section)
                section.npt -= n_start_transition

            seg.section = section
            seg.offset = section.npt
            section.npt += seg.npt

            # create welding region if sample rate increases
            if sample_rate_next != 0 and sample_rate_next > sample_rate:
                # The current section should end before the next segment starts:
                # - subtract any extension into the next segment
                # - align boundary with truncation
                n_pre = int(np.ceil((section.t_end - (seg.t_end - max_pre_start_ns)) * section.sample_rate))
                section.npt -= n_pre
                section.align(extend=False)

                # start new section
                section = RenderSection(sample_rate_next, section.t_end)
                sections.append(section)

                # number of points of segment to be rendered to next section
                n_end_transition = iround((seg.t_end - section.t_start)*sample_rate_next)

                section.npt += n_end_transition

                seg.n_end_transition = n_end_transition
                seg.end_section = section

        # add post stop samples; seg = last segment, section is last section
        n_post = iround(((seg.t_end + max_post_end_ns) - section.t_end) * section.sample_rate)
        if UploadAggregator.verbose:
            logger.debug(f'Post: {n_post}, npt:{section.npt}')
        section.npt += n_post

        # add DC compensation
        compensation_time = self.get_max_compensation_time()
        logger.info(f'DC compensation time: {compensation_time*1e9} ns')
        compensation_npt = int(np.ceil(compensation_time * section.sample_rate * 1e9))
        if compensation_npt > 50_000:
            # More than 50_000 samples? Use new segment with lower sample rate for compensation
            # Upload of 50_000 samples takes ~ 1 ms. It saves upload time to
            # create a new waveform with lower sample rate.

            sample_rate = 1e9 * section.sample_rate * 5_000 / compensation_npt
            # find an existing sample rate
            nice_sample_rates = [1e6, 2e6, 5e6, 1e7, 2e7, 5e7, 1e8, 2e8, 1e9]
            for sr in nice_sample_rates:
                if sample_rate <= sr:
                    sample_rate = sr * 1e-9
                    break
            # create new section
            section.align(extend=True)
            section = RenderSection(sample_rate, section.t_end)
            sections.append(section)
            # calculate npt
            compensation_npt = int(np.ceil(compensation_time * section.sample_rate * 1e9))
            logger.info(f'Added new segment for DC compensation: {int(compensation_time*1e9)} ns, '
                        f'sample_rate: {sr/1e6} MHz, {compensation_npt} Sa')

        job.upload_info.dc_compensation_duration = compensation_npt/section.sample_rate
        section.npt += compensation_npt

        # add at least 1 zero
        section.npt += 1
        section.align(extend=True)
        job.playback_time = section.t_end - sections[0].t_start
        job.n_waveforms = len(sections)
        logger.info(f'Playback time: {job.playback_time} ns')

        if UploadAggregator.verbose:
            for segment in segments:
                logger.info(f'segment: {segment}')
            for section in sections:
                logger.info(f'section: {section}')

    def _generate_upload_wvf(self, job, awg_upload_func):
        segments = self.segments
        sections = job.upload_info.sections
        ref_channel_states = RefChannels(0)

        # loop over all qubit channels to accumulate total phase shift
        for i in range(len(job.sequence)):
            ref_channel_states.start_phases_all.append(dict())
        for channel_name, qubit_channel in self.qubit_channels.items():
            if QsUploader.use_iq_sequencers and channel_name in self.sequencer_channels:
                # skip IQ sequencer channels
                continue
            phase = 0
            for iseg, seg in enumerate(job.sequence):
                ref_channel_states.start_phases_all[iseg][channel_name] = phase
                seg_ch = seg[channel_name]
                phase += seg_ch.get_accumulated_phase(job.index)

        for channel_name, channel_info in self.channels.items():
            if QsUploader.use_iq_sequencers and channel_name in self.sequencer_out_channels:
                # skip IQ sequencer channels
                continue
            if QsUploader.use_baseband_sequencers and channel_name in self.sequencer_channels:
                # skip baseband sequencer channels
                continue

            section = sections[0]
            buffer = np.zeros(section.npt)
            bias_T_compensation_mV = 0

            for iseg, (seg, seg_render) in enumerate(zip(job.sequence, segments)):

                sample_rate = seg_render.sample_rate
                n_delay = iround(channel_info.delay_ns * sample_rate)

                if isinstance(seg, conditional_segment):
                    # logger.debug(f'conditional for {channel_name}')
                    seg_ch = get_conditional_channel(seg, channel_name)
                else:
                    seg_ch = seg[channel_name]
                ref_channel_states.start_time = seg_render.t_start
                ref_channel_states.start_phase = ref_channel_states.start_phases_all[iseg]
                start = time.perf_counter()
                # print(f'start: {channel_name}.{iseg}: {ref_channel_states.start_time}')
                wvf = seg_ch.get_segment(job.index, sample_rate*1e9, ref_channel_states)
                duration = time.perf_counter() - start
                if UploadAggregator.verbose:
                    logger.debug(f'generated [{job.index}]{iseg}:{channel_name} {len(wvf)} Sa, '
                                 f'in {duration*1000:6.3f} ms')

                if len(wvf) != seg_render.npt:
                    logger.warning(f'waveform {iseg}:{channel_name} {len(wvf)} Sa <> sequence length {seg_render.npt}')

                i_start = 0
                if seg_render.start_section:
                    if section != seg_render.start_section:
                        logger.error(f'OOPS section mismatch {iseg}, {channel_name}')

                    # add n_start_transition - n_delay to start_section
#                    n_delay_welding = iround(channel_info.delay_ns * section.sample_rate)
                    t_welding = (section.t_end - seg_render.t_start)
                    i_start = iround(t_welding*sample_rate) - n_delay
                    n_section = (iround(t_welding*section.sample_rate)
                                 + iround(-channel_info.delay_ns * section.sample_rate))

                    if n_section > 0:
                        if iround(n_section*sample_rate/section.sample_rate) >= len(wvf):
                            raise Exception(f'segment {iseg} too short for welding. '
                                            f'(nwelding:{n_section}, len_wvf:{len(wvf)})')

                        isub = [iround(i*sample_rate/section.sample_rate) for i in np.arange(n_section)]
                        welding_samples = np.take(wvf, isub)
                        buffer[-n_section:] = welding_samples

                    bias_T_compensation_mV = self._add_bias_T_compensation(buffer, bias_T_compensation_mV,
                                                                           section.sample_rate, channel_info)
                    self._upload_wvf(job, channel_name, buffer, channel_info.amplitude, channel_info.attenuation,
                                     section.sample_rate, awg_upload_func)

                    section = seg_render.section
                    buffer = np.zeros(section.npt)

                if seg_render.end_section:
                    next_section = seg_render.end_section
                    # add n_end_transition + n_delay to next section. First complete this section
                    n_delay_welding = iround(channel_info.delay_ns * section.sample_rate)
                    t_welding = (seg_render.t_end - next_section.t_start)
                    i_end = len(wvf) - iround(t_welding*sample_rate) + n_delay_welding

                    if i_start != i_end:
                        buffer[-(i_end-i_start):] = wvf[i_start:i_end]

                    bias_T_compensation_mV = self._add_bias_T_compensation(buffer, bias_T_compensation_mV,
                                                                           section.sample_rate, channel_info)
                    self._upload_wvf(job, channel_name, buffer, channel_info.amplitude, channel_info.attenuation,
                                     section.sample_rate, awg_upload_func)

                    section = next_section
                    buffer = np.zeros(section.npt)

                    n_section = (iround(t_welding*section.sample_rate)
                                 + iround(channel_info.delay_ns * section.sample_rate))
                    if iround(n_section*sample_rate/section.sample_rate) >= len(wvf):
                        raise Exception(f'segment {iseg} too short for welding. '
                                        f'(nwelding:{n_section}, len_wvf:{len(wvf)})')

                    isub = [min(len(wvf)-1, i_end + iround(i*sample_rate/section.sample_rate))
                            for i in np.arange(n_section)]
                    welding_samples = np.take(wvf, isub)
                    buffer[:n_section] = welding_samples

                else:
                    if section != seg_render.section:
                        logger.error(f'OOPS-2 section mismatch {iseg}, {channel_name}')
                    offset = seg_render.offset + n_delay
                    buffer[offset+i_start:offset + len(wvf)] = wvf[i_start:]

            if job.neutralize:
                if section != sections[-1]:
                    # DC compensation is in a separate section
                    bias_T_compensation_mV = self._add_bias_T_compensation(buffer, bias_T_compensation_mV,
                                                                           section.sample_rate, channel_info)
                    self._upload_wvf(job, channel_name, buffer, channel_info.amplitude, channel_info.attenuation,
                                     section.sample_rate, awg_upload_func)
                    section = sections[-1]
                    buffer = np.zeros(section.npt)
                    logger.debug(f'DC compensation section with {section.npt} Sa')

                compensation_npt = iround(job.upload_info.dc_compensation_duration * section.sample_rate)

                if compensation_npt > 0 and channel_info.dc_compensation:
                    compensation_voltage = -channel_info.integral * section.sample_rate / compensation_npt * 1e9
                    job.upload_info.dc_compensation_voltages[channel_name] = compensation_voltage
                    buffer[-(compensation_npt+1):-1] = compensation_voltage
                    logger.info(f'DC compensation {channel_name}: '
                                f'{compensation_voltage:6.1f} mV {compensation_npt} Sa')
                else:
                    job.upload_info.dc_compensation_voltages[channel_name] = 0

            bias_T_compensation_mV = self._add_bias_T_compensation(buffer, bias_T_compensation_mV,
                                                                   section.sample_rate, channel_info)
            self._upload_wvf(job, channel_name, buffer, channel_info.amplitude, channel_info.attenuation,
                             section.sample_rate, awg_upload_func)

    def _render_markers(self, job, awg_upload_func):
        for channel_name, marker_channel in self.marker_channels.items():
            if UploadAggregator.verbose:
                logger.debug(f'Marker {channel_name} ({marker_channel.amplitude} mV, {marker_channel.delay:+2.0f} ns)')
            start_stop = []
            if channel_name in self.rf_marker_pulses:
                offset = marker_channel.delay
                for pulse in self.rf_marker_pulses[channel_name]:
                    start_stop.append((offset + pulse.start - marker_channel.setup_ns, +1))
                    start_stop.append((offset + pulse.stop + marker_channel.hold_ns, -1))

            for iseg, (seg, seg_render) in enumerate(zip(job.sequence, self.segments)):
                offset = seg_render.t_start + marker_channel.delay

                if isinstance(seg, conditional_segment):
                    # logger.debug(f'conditional for {channel_name}')
                    seg_ch = get_conditional_channel(seg, channel_name)
                else:
                    seg_ch = seg[channel_name]

                ch_data = seg_ch._get_data_all_at(job.index)

                for pulse in ch_data.my_marker_data:
                    start_stop.append((offset + pulse.start - marker_channel.setup_ns, +1))
                    start_stop.append((offset + pulse.stop + marker_channel.hold_ns, -1))

            m = merge_markers(channel_name, start_stop, min_off_ns=20)
            if marker_channel.channel_number == 0:
                self._upload_fpga_markers(job, marker_channel, m)
            else:
                self._upload_awg_markers(job, marker_channel, m, awg_upload_func)

    def _upload_awg_markers(self, job, marker_channel, m, awg_upload_func):
        sections = job.upload_info.sections
        buffers = [np.zeros(section.npt) for section in sections]
        i_section = 0
        for i in range(0, len(m), 2):
            t_on = m[i][0]
            t_off = m[i+1][0]
            if UploadAggregator.verbose:
                logger.debug(f'Marker: {t_on} - {t_off}')
            # search start section
            while t_on >= sections[i_section].t_end:
                i_section += 1
            section = sections[i_section]
            pt_on = int((t_on - section.t_start) * section.sample_rate)
            if pt_on < 0:
                logger.info('Warning: Marker setup before waveform; aligning with start')
                pt_on = 0
            if t_off < section.t_end:
                pt_off = int((t_off - section.t_start) * section.sample_rate)
                buffers[i_section][pt_on:pt_off] = 1.0
            else:
                buffers[i_section][pt_on:] = 1.0
                i_section += 1
                # search end section
                while t_off >= sections[i_section].t_end:
                    buffers[i_section][:] = 1.0
                    i_section += 1
                section = sections[i_section]
                pt_off = int((t_off - section.t_start) * section.sample_rate)
                buffers[i_section][:pt_off] = 1.0

        for buffer, section in zip(buffers, sections):
            self._upload_wvf(job, marker_channel.name, buffer, 1.0, 1.0, section.sample_rate, awg_upload_func)

    def _upload_fpga_markers(self, job, marker_channel, m):
        table = []
        job.marker_tables[marker_channel.name] = table
        offset = int(self.max_pre_start_ns)
        for i in range(0, len(m), 2):
            # align to marker resolution
            t_on = math.floor((m[i][0] + offset)/10)*10
            t_off = math.ceil((m[i+1][0] + offset)/10)*10
            if UploadAggregator.verbose:
                logger.debug(f'Marker: {t_on} - {t_off}')
            table.append((t_on, t_off))

    def _upload_wvf(self, job, channel_name, waveform, amplitude, attenuation, sample_rate, awg_upload_func):
        # note: numpy inplace multiplication is much faster than standard multiplication
        waveform *= 1/(attenuation * amplitude)
        wave_ref = awg_upload_func(channel_name, waveform)
        job.add_waveform(channel_name, wave_ref, sample_rate*1e9)

    def _preprocess_conditional_segments(self, job):
        self.conditional_segments = [None] * len(job.sequence)
        for iseg, seg in enumerate(job.sequence):
            if isinstance(seg, conditional_segment):
                self.conditional_segments[iseg] = QsConditionalSegment(seg)

    def _get_iq_channel_delay(self, qubit_channel):
        out_channels = qubit_channel.iq_channel.IQ_out_channels
        if len(out_channels) == 1:
            awg_channel_name = out_channels[0].awg_channel_name
            return self.channels[awg_channel_name].delay_ns
        else:
            delays = []
            for i in range(2):
                awg_channel_name = out_channels[i].awg_channel_name
                delays.append(self.channels[awg_channel_name].delay_ns)
            if delays[0] != delays[1]:
                raise Exception(f'I/Q Channel delays must be equal ({qubit_channel.channel_name})')
            return delays[0]

    def _get_iq_channel_attenuation(self, qubit_channel):
        out_channels = qubit_channel.iq_channel.IQ_out_channels

        if len(out_channels) == 1:
            awg_channel_name = out_channels[0].awg_channel_name
            return self.channels[awg_channel_name].attenuation
        else:
            att = [self.channels[output.awg_channel_name].attenuation for output in out_channels]
            if min(att) != max(att):
                raise Exception('Attenuation for IQ output is not equal for channels '
                                f'{[[output.awg_channel_name] for output in out_channels]}')
            return att[0]

    def _generate_sequencer_iq_upload(self, job):
        segments = self.segments

        for channel_name, qubit_channel in self.qubit_channels.items():
            if channel_name not in self.sequencer_channels:
                logger.warning(f'QS driver (M3202A_QS) not loaded for qubit channel {channel_name}')
                continue
            start = time.perf_counter()
            delay = self._get_iq_channel_delay(qubit_channel)
            attenuation = self._get_iq_channel_attenuation(qubit_channel)

            sequencer_offset = self.sequencer_channels[channel_name].sequencer_offset
            # subtract offset, because it's started before 'classical' queued waveform
            t_start = int(-self.max_pre_start_ns - delay) - sequencer_offset

            sequence = IQSequenceBuilder(
                channel_name,
                t_start,
                qubit_channel.iq_channel.LO,
                attenuation=attenuation)
            job.iq_sequences[channel_name] = sequence

            for iseg, (seg, seg_render) in enumerate(zip(job.sequence, segments)):
                if not isinstance(seg, conditional_segment):
                    seg_ch = seg[channel_name]
                    data = seg_ch._get_data_all_at(job.index)
                    entries = data.get_data_elements()
                    # mw_pulses and phase shifts with same start are handled in IQSequenceBuilder
                    for e in entries:
                        t_pulse = seg_render.t_start + e.start
                        if isinstance(e, IQ_data_single):
                            sequence.pulse(t_pulse, e)
                        elif isinstance(e, Chirp):
                            sequence.chirp(t_pulse, e)
                        else:
                            sequence.shift_phase(t_pulse, e.phase_shift)
                else:
                    # logger.debug(f'conditional for {channel_name}:{iseg} start:{seg_render.t_start}')
                    cond_ch = get_conditional_channel(seg, channel_name, sequenced=True, index=job.index)
                    qs_cond = self.conditional_segments[iseg]

                    # A conditional segment can contain a sequence of conditional pulses
                    # that will be triggered individually to optimize waveform memory usage.
                    # Simultaneous pulses are grouped in an conditional instruction.
                    for instr in cond_ch.conditional_instructions:
                        t_instr = seg_render.t_start + instr.start

                        sequence.conditional_pulses(t_instr, seg_render.t_start,
                                                    instr.pulses, qs_cond.order,
                                                    condition_register=3)
            sequence.close()
            duration = time.perf_counter() - start
            if UploadAggregator.verbose:
                logger.debug(f'generated iq sequence {channel_name} {duration*1000:6.3f} ms')

#    def _generate_sequencer_baseband_upload(self, job):
# TODO @@@ baseband pulses
#        for channel_name, channel_info in self.channels.items():
#            if channel_name not in self.sequencer_channels:
#                # skip standard AWG channels
#                continue
#
#            section = sections[0]
#            buffer = np.zeros(section.npt)
#
#            for iseg,(seg,seg_render) in enumerate(zip(job.sequence,segments)):
#
#                sample_rate = seg_render.sample_rate
#                n_delay = iround(channel_info.delay_ns * sample_rate)
#
#                seg_ch = getattr(seg, channel_name)
#                start = time.perf_counter()
#                wvf = seg_ch.get_segment(job.index, sample_rate*1e9)
#                duration = time.perf_counter() - start

    def _check_hvi_triggers(self, hvi_params):
        for name in hvi_params:
            if name.startswith('dig_wait') or name.startswith('dig_trigger'):
                raise Exception(f"digitizer triggering with '{name}' is not supported anymore")

    def _generate_digitizer_triggers(self, job):
        trigger_channels = defaultdict(list)
        job.n_acq_samples = defaultdict(int)
        job.t_measure = {}

        self._check_hvi_triggers(job.schedule_params)

        for ch_name, channel in self.digitizer_channels.items():
            rf_source = channel.rf_source
            if rf_source is not None:
                rf_marker_pulses = []
                # NOTE: this fails when multiple digitizer channels share the same RF marker.
                self.rf_marker_pulses[rf_source.output] = rf_marker_pulses

            offset = int(self.max_pre_start_ns) + channel.delay
            t_end = None
            for iseg, (seg, seg_render) in enumerate(zip(job.sequence, self.segments)):
                if isinstance(seg, conditional_segment):
                    # logger.debug(f'conditional for {channel_name}')
                    seg_ch = get_conditional_channel(seg, ch_name)
                else:
                    seg_ch = seg[ch_name]
                acquisition_data = seg_ch._get_data_all_at(job.index).get_data()
                for acquisition in acquisition_data:
                    if acquisition.n_repeat is not None:
                        raise Exception('Acquisition n_repeat is not supported for Keysight')
                    job.n_acq_samples[ch_name] += 1
                    t = seg_render.t_start + acquisition.start
                    t_measure = (acquisition.t_measure
                                 if acquisition.t_measure is not None
                                 else job.acquisition_conf.t_measure)
                    # if t_measure = -1, then measure till end of sequence. (time trace feature)
                    if t_measure < 0:
                        t_measure = self.segments[-1].t_end - t
                    if ch_name in job.t_measure:
                        if t_measure != job.t_measure[ch_name]:
                            raise Exception(
                                    't_measure must be same for all triggers, '
                                    f'channel:{ch_name}, '
                                    f'{t_measure}!={job.t_measure[ch_name]}')
                    else:
                        job.t_measure[ch_name] = t_measure

                    trigger_channels[t+offset].append(ch_name)
                    t_end = t+t_measure
                    if rf_source is not None and rf_source.mode != 'continuous':
                        rf_marker_pulses.append(RfMarkerPulse(t, t_end))

            if rf_source is not None:
                if rf_source.mode == 'continuous' and t_end is not None:
                    rf_marker_pulses.append(RfMarkerPulse(0, t_end))

                for rf_pulse in rf_marker_pulses:
                    rf_pulse.start += rf_source.delay
                    rf_pulse.stop += rf_source.delay
                    if rf_source.mode in ['pulsed', 'continuous']:
                        rf_pulse.start -= rf_source.startup_time_ns
                        rf_pulse.stop += rf_source.prolongation_ns

        continuous_mode = getattr(job.hw_schedule, 'script_name', '') == 'Continuous'
        if continuous_mode and len(trigger_channels) > 0:
            raise Exception('Digitizer acquisitions are not supported in continuous mode')

        job.digitizer_triggers = dict(sorted(trigger_channels.items()))
        if UploadAggregator.verbose:
            logger.debug(f'digitizer triggers: {job.digitizer_triggers}')

    def _generate_digitizer_sequences(self, job):

        self._check_hvi_triggers(job.schedule_params)
        continuous_mode = getattr(job.hw_schedule, 'script_name', '') == 'Continuous'
        video_mode = 'video_mode_channels' in job.schedule_params

        trigger_channels = defaultdict(list)

        job.n_acq_samples = defaultdict(int)
        job.t_measure = {}

        pxi_triggers = {}
        for seg in job.sequence:
            # assign pxi triggers per conditional segment.
            # WARNING: This gives wrong results with
            #          acquire(ref='m1'); acquire(ref='m2'); Condtional('m1'); Conditional('m2')
            if isinstance(seg, conditional_segment):
                acq_names = get_acquisition_names(seg)
                pxi = 6
                for acq in acq_names:
                    pxi_triggers[acq] = pxi
                    pxi += 1

        logger.debug(f'PXI triggers: {pxi_triggers}')

        segments = self.segments
        for channel_name, channel in self.digitizer_channels.items():
            dig = self.digitizers[channel.module_name]
            if not hasattr(dig, 'get_sequencer'):
                raise Exception(f'QS driver (M3102A_QS) not configured for digitizer {channel.module_name}')
            sequence = AcquisitionSequenceBuilder(channel_name)
            job.digitizer_sequences[channel_name] = sequence
            rf_type = None
            rf_sequence = None
            rf_source = channel.rf_source
            if rf_source is not None:
                if isinstance(rf_source.output, str):
                    rf_type = 'marker'
                    rf_marker_pulses = []
                    # NOTE: this fails when multiple digitizer channels share the same RF marker.
                    self.rf_marker_pulses[rf_source.output] = rf_marker_pulses
                elif channel_name+'_RF' in self.rf_sequencers:
                    rf_channel_name = channel_name+'_RF'
                    rf_type = 'generator'
                    sequencer_offset = self.rf_sequencers[rf_channel_name].sequencer_offset
                    rf_sequence = RFSequenceBuilder(rf_channel_name, rf_source,
                                                    int(self.max_pre_start_ns+sequencer_offset))
                    job.rf_sequences[rf_channel_name] = rf_sequence

            t_end = None
            for iseg, (seg, seg_render) in enumerate(zip(job.sequence, segments)):
                offset = channel.delay + int(self.max_pre_start_ns)
                if isinstance(seg, conditional_segment):
                    # logger.debug(f'conditional for {channel_name}')
                    # TODO @@@ lookup acquisitions and set pxi trigger.
                    seg_ch = get_conditional_channel(seg, channel_name)
                else:
                    seg_ch = seg[channel_name]
                acquisition_data = seg_ch._get_data_all_at(job.index).get_data()
                for acquisition in acquisition_data:
                    if continuous_mode:
                        raise Exception('Digitizer acquisitions are not supported in continuous mode')
                    t = seg_render.t_start + acquisition.start
                    if acquisition.t_measure is not None:
                        t_measure = acquisition.t_measure
                    else:
                        t_measure = job.acquisition_conf.t_measure
                    # if t_measure = -1, then measure till end of sequence. (time trace feature)
                    if t_measure < 0:
                        t_measure = self.segments[-1].t_end - t
                    if job.acquisition_conf.sample_rate is not None:
                        period_ns = iround(1e8/job.acquisition_conf.sample_rate) * 10
                        n_cycles = int(t_measure / period_ns)
                        t_integrate = period_ns
                    else:
                        t_integrate = t_measure
                        n_cycles = 1
                    job.n_acq_samples[channel_name] += n_cycles
                    pxi_trigger = pxi_triggers.get(str(acquisition.ref), None)
                    sequence.acquire(t + offset, t_integrate, n_cycles,
                                     threshold=acquisition.threshold,
                                     pxi_trigger=pxi_trigger)

                    job.t_measure[channel_name] = t_measure
                    if UploadAggregator.verbose:
                        logger.debug(f'Acq: {acquisition.ref}: {pxi_trigger}')
                    t_end = t+t_measure
                    if rf_source is not None and rf_source.mode != 'continuous':
                        if rf_type == 'marker':
                            rf_marker_pulses.append(RfMarkerPulse(t, t_end))
                        elif rf_type == 'generator':
                            rf_sequence.enable(t, t_end)
                    trigger_channels[t+offset].append(channel_name)

            if rf_source is not None:
                if rf_source.mode == 'continuous' and t_end is not None:
                    if rf_type == 'marker':
                        rf_marker_pulses.append(RfMarkerPulse(0, t_end))
                    elif rf_type == 'generator':
                        rf_sequence.enable(t, t_end)

                if rf_type == 'marker':
                    for rf_pulse in rf_marker_pulses:
                        rf_pulse.start += rf_source.delay
                        rf_pulse.stop += rf_source.delay
                        if rf_source.mode in ['pulsed', 'continuous']:
                            rf_pulse.start -= rf_source.startup_time_ns
                            rf_pulse.stop += rf_source.prolongation_ns

            if video_mode and rf_sequence is not None:
                dig_name = channel.module_name
                video_mode_channels = job.schedule_params['video_mode_channels']
                if (dig_name in video_mode_channels
                        and (set(channel.channel_numbers) & set(video_mode_channels[dig_name]))):
                    rf_sequence.enable(0, job.upload_info.sections[-1].t_end)

            sequence.close()
        # Only used for oscillators
        job.digitizer_triggers = dict(sorted(trigger_channels.items()))

    def upload_job(self, job, awg_upload_func):

        job.upload_info = JobUploadInfo()
        job.marker_tables = {}
        job.iq_sequences = {}
        job.digitizer_sequences = {}
        job.digitizer_triggers = {}
        job.rf_sequences = {}
        self.rf_marker_pulses = {}

        self._integrate(job)
        self._generate_sections(job)
        self._preprocess_conditional_segments(job)
        self._generate_upload_wvf(job, awg_upload_func)
        if QsUploader.use_iq_sequencers:
            self._generate_sequencer_iq_upload(job)
        # if QsUploader.use_baseband_sequencers:
        #     self._generate_sequencer_baseband_upload(job)
        if QsUploader.use_digitizer_sequencers:
            self._generate_digitizer_sequences(job)
        else:
            self._generate_digitizer_triggers(job)
        self._render_markers(job, awg_upload_func)

    def get_max_compensation_time(self):
        '''
        generate a DC compensation of the pulse.
        As usuallly we put capacitors in between the AWG and the gate on the sample, you need to correct
        for the fact that the low fequencies are not present in your transfer function.
        This can be done simply by making the total integral of your function 0.

        Args:
            sample_rate (float) : rate at which the AWG runs.
        '''
        return max(self.get_compensation_time(channel_info) for channel_info in self.channels.values())

    def get_compensation_time(self, channel_info):
        '''
        return the minimal compensation time that is needed.
        Returns:
            compensation_time : minimal duration that is needed for the voltage compensation
        '''
        if not channel_info.dc_compensation:
            return 0

        if channel_info.integral <= 0:
            result = -channel_info.integral / channel_info.dc_compensation_max
        else:
            result = -channel_info.integral / channel_info.dc_compensation_min
        return result

    def _add_bias_T_compensation(self, buffer, bias_T_compensation_mV, sample_rate, channel_info):

        if channel_info.bias_T_RC_time:
            compensation_factor = 1 / (sample_rate * 1e9 * channel_info.bias_T_RC_time)
            compensation = np.cumsum(buffer) * compensation_factor + bias_T_compensation_mV
            bias_T_compensation_mV = compensation[-1]
            if UploadAggregator.verbose:
                logger.info(f'bias-T compensation  min:{np.min(compensation):5.1f} max:{np.max(compensation):5.1f} mV')
            buffer += compensation

        return bias_T_compensation_mV
