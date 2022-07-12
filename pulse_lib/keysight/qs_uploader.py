import time
import numpy as np
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional

from .sequencer_device import add_sequencers
from .qs_conditional import get_conditional_channel, get_acquisition_names, QsConditionalSegment
from .qs_sequence import AcquisitionSequenceBuilder, IQSequenceBuilder, SequenceConditionalEntry

from pulse_lib.segments.data_classes.data_IQ import IQ_data_single
from pulse_lib.segments.conditional_segment import conditional_segment
from pulse_lib.tests.mock_m3202a_qs import AwgInstruction, AwgConditionalInstruction
from pulse_lib.tests.mock_m3102a_qs import DigitizerInstruction
from pulse_lib.segments.utility.rounding import iround


class AwgConfig:
    MAX_AMPLITUDE = 1500 # mV
    ALIGNMENT = 10 # waveform must be multiple 10 bytes


class QsUploader:
    verbose = True
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

        add_sequencers(self, awg_devices, awg_channels, IQ_channels)

        self.release_all_awg_memory()

        self._config_marker_channels()

    @property
    def supports_conditionals(self):
        return True

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


    def get_effective_sample_rate(self, sample_rate):
        """
        Returns the sample rate that will be used by the Keysight AWG.
        This is the a rate >= requested sample rate.
        """
        awg = list(self.AWGs.values())[0]
        return awg.convert_prescaler_to_sample_rate(awg.convert_sample_rate_to_prescaler(sample_rate))

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
            job (upload_job) : upload_job object that defines what needs to be uploaded and possible post processing of the waveforms (if needed)
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

        aggregator = UploadAggregator(self.AWGs, self.awg_channels, self.marker_channels, self.digitizer_channels,
                                      self.qubit_channels, self.sequencer_channels, self.sequencer_out_channels)

        aggregator.upload_job(job, self.__upload_to_awg) # @@@ TODO split generation and upload

        self.jobs.append(job)

        duration = time.perf_counter() - start
        logging.debug(f'generated upload data ({duration*1000:6.3f} ms)')


    def __upload_to_awg(self, channel_name, waveform):
#        vmin = waveform.min()
#        vmax = waveform.max()
#        length = len(waveform)
#        logging.debug(f'{channel_name}: V({vmin*1000:6.3f}, {vmax*1000:6.3f}) {length}')
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
        if not channel_name in self.marker_channels:
            raise Exception(f'Channel {channel_name} not found in configuration')
        marker_channel = self.marker_channels[channel_name]
        awg_name = marker_channel.module_name
        awg = self.AWGs[awg_name]
        awg.load_marker_table(table)
        logging.debug(f'marker for {channel_name} loaded in {(time.perf_counter()-start)*1000:4.2f} ms')

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

        logging.error(f'Job not found for index {index} of seq {seq_id}')
        raise ValueError(f'Sequence with id {seq_id}, index {index} not placed for upload .. . Always make sure to first upload your segment and then do the playback.')


    def play(self, seq_id, index, release_job = True):
        """
        start playback of a sequence that has been uploaded.
        Args:
            seq_id (uuid) : id of the sequence
            index (tuple) : index that has to be played
            release_job (bool) : release memory on AWG after done.
        """

        job =  self.__get_job(seq_id, index)
        self.wait_until_AWG_idle()

        for channel_name, marker_table in job.marker_tables.items():
            self.__upload_markers(channel_name, marker_table)

        for awg_channel in self.awg_channels.values():
            awg_name = awg_channel.awg_name
            channel_number = awg_channel.channel_number
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

                start_delay = 0 # no start delay
                trigger_mode = 1 # software/HVI trigger
                cycles = 1
                for queue_item in queue:
                    prescaler = awg.convert_sample_rate_to_prescaler(queue_item.sample_rate)
                    awg.awg_queue_waveform(
                            channel_number, queue_item.wave_reference,
                            trigger_mode, start_delay, cycles, prescaler)
                    trigger_mode = 0 # Auto tigger -- next waveform will play automatically.
            except:
                raise Exception(f'Play failed on channel {channel_name}')


        # set offset for IQ channels
        for channel_name, awg_channel in self.awg_channels.items():
            if channel_name not in job.channel_queues:
                awg_name = awg_channel.awg_name
                channel_number = awg_channel.channel_number
                offset = awg_channel.offset
                awg = self.AWGs[awg_name]
                awg.set_channel_offset(offset/1000, channel_number)

        start = time.perf_counter()
        for awg_sequencer in self.sequencer_channels.values():
            awg = self.AWGs[awg_sequencer.module_name]
            channel_name = awg_sequencer.channel_name
            seq = awg.get_sequencer(awg_sequencer.sequencer_index)
            seq.flush_waveforms()
            schedule = []
            if channel_name in job.iq_sequences:
                t1 = time.perf_counter()
                sequence = job.iq_sequences[awg_sequencer.channel_name]

                # TODO cleanup frequency update hack
                qubit_channel = self.qubit_channels[awg_sequencer.channel_name]
                seq._frequency = qubit_channel.reference_frequency - qubit_channel.iq_channel.LO

                # @@@ IQSequence.upload() OR Sequence.upload()
                for number,wvf in enumerate(sequence.waveforms):
                    seq.upload_waveform(number, wvf.offset, wvf.duration,
                                        wvf.amplitude, wvf.am_envelope,
                                        wvf.frequency, wvf.pm_envelope,
                                        wvf.prephase, wvf.postphase, wvf.restore_frequency)

                t2 = time.perf_counter()
                for i,entry in enumerate(sequence.sequence):
                    if isinstance(entry, SequenceConditionalEntry):
                        schedule.append(AwgConditionalInstruction(i, entry.time_after,
                                                                  wave_numbers=entry.waveform_indices,
                                                                  condition_register=entry.cr))
                    else:
                        schedule.append(AwgInstruction(i, entry.time_after, wave_number=entry.waveform_index))
                t3 = time.perf_counter()
                logging.debug(f'{awg_sequencer.channel_name} create waves:{(t2-t1)*1000:6.3f}, seq:{(t3-t2)*1000:6.3f} ms')
            seq.load_schedule(schedule)
        logging.debug(f'loaded awg sequences in {(time.perf_counter() - start)*1000:6.3f} ms')

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
                    for i,entry in enumerate(sequence.sequence):
                        schedule.append(DigitizerInstruction(i, entry.time_after,
                                                             t_measure=entry.t_measure,
                                                             n_cycles=entry.n_cycles,
                                                             measurement_id=entry.measurement_id,
                                                             pxi=entry.pxi_trigger,
                                                             threshold=entry.threshold))
                    seq.load_schedule(schedule)

        logging.debug(f'loaded dig sequences in {(time.perf_counter() - start)*1000:6.3f} ms')

        # start hvi (start function loads schedule if not yet loaded)
        acquire_triggers = {f'dig_trigger_{i+1}':t for i,t in enumerate(job.digitizer_triggers)}
        trigger_channels = {f'dig_trigger_channels_{dig_name}':triggers
                            for dig_name, triggers in job.digitizer_trigger_channels.items()}
        schedule_params = job.schedule_params.copy()
        schedule_params.update(acquire_triggers)
        schedule_params.update(trigger_channels)
        job.hw_schedule.set_configuration(schedule_params, job.n_waveforms)
        job.hw_schedule.start(job.playback_time, job.n_rep, schedule_params)

        if release_job:
            job.release()


    def release_memory(self, seq_id=None, index=None):
        """
        Release job memory for `seq_id` and `index`.
        Args:
            seq_id (uuid) : id of the sequence. if None release all
            index (tuple) : index that has to be released; if None release all.
        """
        for job in self.jobs:
            if (seq_id is None
                or (job.seq_id == seq_id and (index is None or job.index == index))):
                job.release()

    def release_all_awg_memory(self):
        for awg in self.AWGs.values():
            if hasattr(awg, 'release_waveform_memory'):
                awg.release_waveform_memory()
            else:
                print('Update M3202A driver')

    def release_jobs(self):
        for job in self.jobs:
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
class AwgQueueItem:
    wave_reference: object
    sample_rate: float


class Job(object):
    """docstring for upload_job"""
    def __init__(self, job_list, sequence, index, seq_id, n_rep, sample_rate, neutralize=True, priority=0):
        '''
        Args:
            job_list (list): list with all jobs.
            sequence (list of list): list with list of the sequence
            index (tuple) : index that needs to be uploaded
            seq_id (uuid) : if of the sequence
            n_rep (int) : number of repetitions of this sequence.
            sample_rate (float) : sample rate
            neutralize (bool) : place a neutralizing segment at the end of the upload
            priority (int) : priority of the job (the higher one will be excuted first)
        '''
        self.job_list = job_list
        self.sequence = sequence
        self.seq_id = seq_id
        self.index = index
        self.n_rep = n_rep
        self.default_sample_rate = sample_rate
        self.neutralize = neutralize
        self.priority = priority
        self.playback_time = 0 #total playtime of the waveform

        self.released = False

        self.channel_queues = dict()
        self.hw_schedule = None
        logging.debug(f'new job {seq_id}-{index}')


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

    def add_waveform(self, channel_name, wave_ref, sample_rate):
        if channel_name not in self.channel_queues:
            self.channel_queues[channel_name] = []

        self.channel_queues[channel_name].append(AwgQueueItem(wave_ref, sample_rate))


    def release(self):
        if self.released:
            logging.warning(f'job {self.seq_id}-{self.index} already released')
            return

        self.upload_info = None
        logging.debug(f'release job {self.seq_id}-{self.index}')
        self.released = True

        for channel_name, queue in self.channel_queues.items():
            for queue_item in queue:
                queue_item.wave_reference.release()

        if self in self.job_list:
            self.job_list.remove(self)


    def __del__(self):
        if not self.released:
            logging.warn(f'Job {self.seq_id}-{self.index} was not released. '
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
    t_start: float # can be negative for negative channel delays
    npt: int = 0

    @property
    def t_end(self):
        return self.t_start + self.npt / self.sample_rate

    def align(self, extend):
        if extend:
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
    start_phase: Dict[str,float] = field(default_factory=dict)
    start_phases_all: List[Dict[str,float]] = field(default_factory=list)


class UploadAggregator:
    verbose = False

    def __init__(self, AWGs, awg_channels, marker_channels, digitizer_channels,
                 qubit_channels, sequencer_channels, sequencer_out_channels):
        self.AWGs = AWGs
        self.npt = 0
        self.marker_channels = marker_channels
        self.digitizer_channels = digitizer_channels
        self.sequencer_channels = sequencer_channels
        self.sequencer_out_channels = sequencer_out_channels
        self.qubit_channels = qubit_channels

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

        self.max_pre_start_ns = -min(0, *delays)
        self.max_post_end_ns = max(0, *delays)


    def _integrate(self, job):

        if not job.neutralize:
            return

        for iseg,seg in enumerate(job.sequence):
            sample_rate = seg.sample_rate if seg.sample_rate is not None else job.default_sample_rate

            for channel_name, channel_info in self.channels.items():
                if iseg == 0:
                    channel_info.integral = 0

                if channel_info.dc_compensation:
                    if isinstance(seg, conditional_segment):
                        seg_ch = get_conditional_channel(seg, channel_name)
                    else:
                        seg_ch = seg[channel_name]
                    channel_info.integral += seg_ch.integrate(job.index, sample_rate)
                    logging.debug(f'Integral seg:{iseg} {channel_name} integral:{channel_info.integral}')


    def _generate_sections(self, job):
        max_pre_start_ns = self.max_pre_start_ns
        max_post_end_ns = self.max_post_end_ns

        self.segments = []
        segments = self.segments
        t_start = 0
        for seg in job.sequence:
            # work with sample rate in GSa/s
            sample_rate = (seg.sample_rate if seg.sample_rate is not None else job.default_sample_rate) * 1e-9
            duration = seg.get_total_time(job.index)
            npt =  iround(duration * sample_rate)
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

        for iseg,seg in enumerate(segments):
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
        section.npt += n_post

        # add DC compensation
        compensation_time = self.get_max_compensation_time()
        logging.debug(f'DC compensation time: {compensation_time*1e9} ns')
        compensation_npt = int(np.ceil(compensation_time * section.sample_rate * 1e9))
        if compensation_npt > 50_000:
            # more than 50_000 samples? Use new segment with lower sample rate for compensation

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
            logging.info(f'Added new segment for DC compensation: {int(compensation_time*1e9)} ns, '
                         f'sample_rate: {sr/1e6} MHz, {compensation_npt} Sa')

        job.upload_info.dc_compensation_duration = compensation_npt/section.sample_rate
        section.npt += compensation_npt

        # add at least 1 zero
        section.npt += 1
        section.align(extend=True)
        job.playback_time = section.t_end - sections[0].t_start
        job.n_waveforms = len(sections)
        logging.debug(f'Playback time: {job.playback_time} ns')

        if UploadAggregator.verbose:
            for segment in segments:
                logging.info(f'segment: {segment}')
            for section in sections:
                logging.info(f'section: {section}')


    def _generate_upload_wvf(self, job, awg_upload_func):
        segments = self.segments
        sections = job.upload_info.sections
        ref_channel_states = RefChannels(0)

        # loop over all qubit channels to accumulate total phase shift
        for i in range(len(job.sequence)):
            ref_channel_states.start_phases_all.append(dict())
        for channel_name, qubit_channel in self.qubit_channels.items():
            if (QsUploader.use_iq_sequencers
                and channel_name in self.sequencer_channels):
                # skip IQ sequencer channels
                continue
            phase = 0
            for iseg,seg in enumerate(job.sequence):
                ref_channel_states.start_phases_all[iseg][channel_name] = phase
                seg_ch = seg[channel_name]
                phase += seg_ch.get_accumulated_phase(job.index)

        for channel_name, channel_info in self.channels.items():
            if (QsUploader.use_iq_sequencers
                and channel_name in self.sequencer_out_channels):
                # skip IQ sequencer channels
                continue
            if (QsUploader.use_baseband_sequencers
                and channel_name in self.sequencer_channels):
                # skip baseband sequencer channels
                continue

            section = sections[0]
            buffer = np.zeros(section.npt)
            bias_T_compensation_mV = 0

            for iseg,(seg,seg_render) in enumerate(zip(job.sequence,segments)):

                sample_rate = seg_render.sample_rate
                n_delay = iround(channel_info.delay_ns * sample_rate)

                if isinstance(seg, conditional_segment):
                    logging.debug(f'conditional for {channel_name}')
                    seg_ch = get_conditional_channel(seg, channel_name)
                else:
                    seg_ch = seg[channel_name]
                ref_channel_states.start_time = seg_render.t_start
                ref_channel_states.start_phase = ref_channel_states.start_phases_all[iseg]
                start = time.perf_counter()
                #print(f'start: {channel_name}.{iseg}: {ref_channel_states.start_time}')
                wvf = seg_ch.get_segment(job.index, sample_rate*1e9, ref_channel_states)
                duration = time.perf_counter() - start
                logging.debug(f'generated [{job.index}]{iseg}:{channel_name} {len(wvf)} Sa, in {duration*1000:6.3f} ms')

                if len(wvf) != seg_render.npt:
                    logging.warn(f'waveform {iseg}:{channel_name} {len(wvf)} Sa <> sequence length {seg_render.npt}')

                i_start = 0
                if seg_render.start_section:
                    if section != seg_render.start_section:
                        logging.error(f'OOPS section mismatch {iseg}, {channel_name}')

                    # add n_start_transition - n_delay to start_section
#                    n_delay_welding = iround(channel_info.delay_ns * section.sample_rate)
                    t_welding = (section.t_end - seg_render.t_start)
                    i_start = iround(t_welding*sample_rate) - n_delay
                    n_section = iround(t_welding*section.sample_rate) + iround(-channel_info.delay_ns * section.sample_rate)

                    if n_section > 0:
                        if iround(n_section*sample_rate/section.sample_rate) >= len(wvf):
                            raise Exception(f'segment {iseg} too short for welding. (nwelding:{n_section}, len_wvf:{len(wvf)})')

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

                    n_section = iround(t_welding*section.sample_rate) + iround(channel_info.delay_ns * section.sample_rate)
                    if iround(n_section*sample_rate/section.sample_rate) >= len(wvf):
                        raise Exception(f'segment {iseg} too short for welding. (nwelding:{n_section}, len_wvf:{len(wvf)})')

                    isub = [min(len(wvf)-1, i_end + iround(i*sample_rate/section.sample_rate)) for i in np.arange(n_section)]
                    welding_samples = np.take(wvf, isub)
                    buffer[:n_section] = welding_samples

                else:
                    if section != seg_render.section:
                        logging.error(f'OOPS-2 section mismatch {iseg}, {channel_name}')
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
                    logging.info(f'DC compensation section with {section.npt} Sa')

                compensation_npt = iround(job.upload_info.dc_compensation_duration * section.sample_rate)

                if compensation_npt > 0 and channel_info.dc_compensation:
                    compensation_voltage = -channel_info.integral * section.sample_rate / compensation_npt * 1e9
                    job.upload_info.dc_compensation_voltages[channel_name] = compensation_voltage
                    buffer[-(compensation_npt+1):-1] = compensation_voltage
                    logging.debug(f'DC compensation {channel_name}: {compensation_voltage:6.1f} mV {compensation_npt} Sa')
                else:
                    job.upload_info.dc_compensation_voltages[channel_name] = 0

            bias_T_compensation_mV = self._add_bias_T_compensation(buffer, bias_T_compensation_mV,
                                                                   section.sample_rate, channel_info)
            self._upload_wvf(job, channel_name, buffer, channel_info.amplitude, channel_info.attenuation,
                             section.sample_rate, awg_upload_func)

    def _render_markers(self, job, awg_upload_func):
        for channel_name, marker_channel in self.marker_channels.items():
            logging.debug(f'Marker: {channel_name} ({marker_channel.amplitude} mV, {marker_channel.delay:+2.0f} ns)')
            start_stop = []
            for iseg, (seg, seg_render) in enumerate(zip(job.sequence, self.segments)):

                if isinstance(seg, conditional_segment):
                    logging.debug(f'conditional for {channel_name}')
                    seg_ch = get_conditional_channel(seg, channel_name)
                else:
                    seg_ch = seg[channel_name]

                ch_data = seg_ch._get_data_all_at(job.index)

                for pulse in ch_data.my_marker_data:
                    offset = seg_render.t_start + marker_channel.delay
                    start_stop.append((offset + pulse.start - marker_channel.setup_ns, +1))
                    start_stop.append((offset + pulse.stop + marker_channel.hold_ns, -1))

            if len(start_stop) > 0:
                m = np.array(start_stop)
                ind = np.argsort(m[:,0])
                m = m[ind]
            else:
                m = []

            # @@@ consolidate on/off
            if marker_channel.channel_number == 0:
                self._upload_fpga_markers(job, marker_channel, m)
            else:
                self._upload_awg_markers(job, marker_channel, m, awg_upload_func)

    def _upload_awg_markers(self, job, marker_channel, m, awg_upload_func):
        # TODO @@@ : round section length to 100 ns and use max 1e8 Sa/s rendering?
        sections = job.upload_info.sections
        buffers = [np.zeros(section.npt) for section in sections]
        i_section = 0
        s = 0
        t_on = 0
        for on_off in m:
            s += on_off[1]
            if s < 0:
                logging.error(f'Marker error {marker_channel.name} {on_off}')
            if s == 1 and on_off[1] == 1:
                t_on = on_off[0]
            if s == 0 and on_off[1] == -1:
                t_off = on_off[0]
                logging.debug(f'Marker: {t_on} - {t_off}')
                # search start section
                while t_on >= sections[i_section].t_end:
                    i_section += 1
                section = sections[i_section]
                pt_on = int((t_on - section.t_start) * section.sample_rate)
                if pt_on < 0:
                    logging.info(f'Warning: Marker setup before waveform; aligning with start')
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
        s = 0
        t_on = 0
        for on_off in m:
            s += on_off[1]
            if s < 0:
                logging.error(f'Marker error {marker_channel.name} {on_off}')
            if s == 1 and on_off[1] == 1:
                t_on = int(on_off[0])
            if s == 0 and on_off[1] == -1:
                t_off = int(on_off[0])
                logging.debug(f'Marker: {t_on} - {t_off}')
                table.append((t_on + offset, t_off + offset))

    def _upload_wvf(self, job, channel_name, waveform, amplitude, attenuation, sample_rate, awg_upload_func):
        # note: numpy inplace multiplication is much faster than standard multiplication
        waveform *= 1/(attenuation * amplitude)
        wave_ref = awg_upload_func(channel_name, waveform)
        job.add_waveform(channel_name, wave_ref, sample_rate*1e9)

    def _preprocess_conditional_segments(self, job):
        self.conditional_segments = [None] * len(job.sequence)
        for iseg,seg in enumerate(job.sequence):
            if isinstance(seg, conditional_segment):
                self.conditional_segments[iseg] = QsConditionalSegment(seg)

    def _generate_sequencer_iq_upload(self, job):
        segments = self.segments

        for channel_name, qubit_channel in self.qubit_channels.items():
            start = time.perf_counter()
            delays = []
            for i in range(2):
                awg_channel_name = qubit_channel.iq_channel.IQ_out_channels[i].awg_channel_name
                delays.append(self.channels[awg_channel_name].delay_ns)
            if delays[0] != delays[1]:
                raise Exception(f'I/Q Channel delays must be equal ({channel_name})')

            # TODO improve for alignment on 1 ns. -> set channel delay in FPGA
            # subtract 10 ns, because it's started 10 ns before 'classical' queued waveform
            t_start = int((-self.max_pre_start_ns - delays[0]) / 5) * 5 -10

            sequence = IQSequenceBuilder(channel_name, t_start,
                                         qubit_channel.iq_channel.LO)
            job.iq_sequences[channel_name] = sequence


            for iseg,(seg,seg_render) in enumerate(zip(job.sequence,segments)):
                if not isinstance(seg, conditional_segment):
                    seg_ch = seg[channel_name]
                    data = seg_ch._get_data_all_at(job.index)
                    # TODO @@@ Fix mw_pulses and phase shifts with same start !
                    mw_data = {pulse.start:pulse for pulse in data.MW_pulse_data}
                    mw_data.update({ps.time:ps for ps in data.phase_shifts if ps.phase_shift != 0})

                    for mw_time,mw_entry in sorted(mw_data.items()):
                        t_pulse = seg_render.t_start + mw_time

                        if isinstance(mw_entry, IQ_data_single):
                            sequence.pulse(t_pulse, mw_entry)
                        else:
                            sequence.shift_phase(t_pulse, mw_entry.phase_shift)
                else:
                    logging.debug(f'conditional for {channel_name}:{iseg} start:{seg_render.t_start}')
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
            logging.debug(f'generated iq sequence {channel_name} {duration*1000:6.3f} ms')

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

    def _generate_digitizer_triggers(self, job):
        offset = int(self.max_pre_start_ns)
        trigger_channels = {}
        digitizer_trigger_channels = {}
        has_HVI_triggers = False

        for name, value in job.schedule_params.items():
            if name.startswith('dig_trigger_') or name.startswith('dig_wait'):
                has_HVI_triggers = True

        # TODO @@@: cleanup this messy code.
        for channel_name, channel in self.digitizer_channels.items():
            for iseg, (seg, seg_render) in enumerate(zip(job.sequence, self.segments)):
                if isinstance(seg, conditional_segment):
                    logging.debug(f'conditional for {channel_name}')
                    seg_ch = get_conditional_channel(seg, channel_name)
                else:
                    seg_ch = seg[channel_name]
                acquisition_data = seg_ch._get_data_all_at(job.index).get_data()
                for acquisition in acquisition_data:
                    if has_HVI_triggers:
                        raise Exception('Cannot combine HVI digitizer triggers with acquisition() calls')
                    t = seg_render.t_start + acquisition.start + offset
                    for ch in channel.channel_numbers:
                        trigger_channels.setdefault(t, []).append((channel.module_name, ch))
                    # set empty list. Fill later after sorting all triggers
                    digitizer_trigger_channels[channel.module_name] = []

        job.digitizer_triggers = list(trigger_channels.keys())
        job.digitizer_triggers.sort()
        for name, triggers in digitizer_trigger_channels.items():
            for trigger in job.digitizer_triggers:
                all_channels = trigger_channels[trigger]
                triggers.append([nr for module_name, nr in all_channels if module_name == name])

        job.digitizer_trigger_channels = digitizer_trigger_channels
        logging.info(f'digitizer triggers: {job.digitizer_triggers}')

    def _generate_digitizer_sequences(self, job):

        for name, value in job.schedule_params.items():
            if name.startswith('dig_trigger_') or name.startswith('dig_wait'):
                raise Exception('HVI triggers not supported with QS')

        pxi_triggers = {}
        for seg in job.sequence:
            if isinstance(seg, conditional_segment):
                acq_names = get_acquisition_names(seg)
                pxi = 6
                for acq in acq_names:
                    pxi_triggers[acq] = pxi
                    pxi += 1

        logging.debug(f'PXI triggers: {pxi_triggers}')

        offset = int(self.max_pre_start_ns)
        segments = self.segments
        for channel_name, channel in self.digitizer_channels.items():
            t_start = -offset
            sequence = AcquisitionSequenceBuilder(channel_name, t_start)
            job.digitizer_sequences[channel_name] = sequence

            for iseg, (seg, seg_render) in enumerate(zip(job.sequence, segments)):
                if isinstance(seg, conditional_segment):
                    logging.debug(f'conditional for {channel_name}')
                    # TODO @@@ lookup acquisitions and set pxi trigger.
                    seg_ch = get_conditional_channel(seg, channel_name)
                else:
                    seg_ch = seg[channel_name]
                acquisition_data = seg_ch._get_data_all_at(job.index).get_data()
                for acquisition in acquisition_data:
                    t_acq = seg_render.t_start + acquisition.start

                    if acquisition.t_measure is not None:
                        t_measure = acquisition.t_measure
                    else:
                        t_measure = job.acquisition_conf.t_measure
                    if job.acquisition_conf.sample_rate is not None:
                        period_ns = iround(1e8/job.acquisition_conf.sample_rate) * 10
                        n_cycles = int(t_measure / period_ns)
                        t_integrate = period_ns
                    else:
                        t_integrate = t_measure
                        n_cycles = 1
                    pxi_trigger = pxi_triggers.get(str(acquisition.ref), None)
                    sequence.acquire(t_acq, t_integrate, n_cycles,
                                     threshold=acquisition.threshold,
                                     pxi_trigger=pxi_trigger)

                    logging.debug(f'Acq: {acquisition.ref}: {pxi_trigger}')
            sequence.close()


    def upload_job(self, job, awg_upload_func):

        job.upload_info = JobUploadInfo()
        job.marker_tables = {} # @@@ add to upload_info?
        job.iq_sequences = {}
        job.digitizer_sequences = {}
        job.digitizer_triggers = {}
        job.digitizer_trigger_channels = {}

        self._integrate(job)

        self._generate_sections(job)

        self._preprocess_conditional_segments(job)

        self._generate_upload_wvf(job, awg_upload_func)

        self._render_markers(job, awg_upload_func)

        if QsUploader.use_iq_sequencers:
            self._generate_sequencer_iq_upload(job)

#        if QsUploader.use_baseband_sequencers:
#            self._generate_sequencer_baseband_upload(job)

        if QsUploader.use_digitizer_sequencers:
            self._generate_digitizer_sequences(job)
        else:
            self._generate_digitizer_triggers(job)


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
            logging.info(f'bias-T compensation  min:{np.min(compensation):5.1f} max:{np.max(compensation):5.1f} mV')
            buffer += compensation

        return bias_T_compensation_mV

