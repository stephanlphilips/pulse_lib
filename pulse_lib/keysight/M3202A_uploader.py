import time
import numpy as np
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional

from keysightSD1 import SD_TriggerExternalSources, SD_FpgaTriggerDirection, SD_TriggerPolarity


class AwgConfig:
    MAX_AMPLITUDE = 1500 # mV
    ALIGNMENT = 10 # waveform must be multiple 10 bytes


class M3202A_Uploader:

    verbose = False

    def __init__(self, AWGs, awg_channels, marker_channels, qubit_channels):
        '''
        Initialize the keysight uploader.
        Args:
            AWGs (dict<awg_name,QcodesIntrument>): list with AWG's
            awg_channels Dict[name, awg_channel]: channel names and properties
            marker_channels: Dict[name, marker_channel]: dict with names and properties
            qubit_channels: Dict[name, qubit_channel]: dict with names and properties
        Returns:
            None
        '''
        self.AWGs = AWGs

        self.awg_channels = awg_channels
        self.marker_channels = marker_channels
        self.qubit_channels = qubit_channels

        self.jobs = []
        # hvi is used by scheduler to check whether another hvi must be loaded.
        self.hvi = None

        self._config_marker_channels()

    def _config_marker_channels(self):
        for channel in self.marker_channels.values():
            awg_name = channel.module_name
            awg = self.AWGs[awg_name]
            if channel.channel_number == 0:
                awg.config_fpga_trigger(
                        SD_TriggerExternalSources.TRIGGER_EXTERN,
                        SD_FpgaTriggerDirection.INOUT,
                        SD_TriggerPolarity.ACTIVE_HIGH if not channel.invert else SD_TriggerPolarity.ACTIVE_LOW
                        )
            else:
                offset = 0
                amplitude = AwgConfig.MAX_AMPLITUDE/1000
                if channel.invert:
                    offset = channel.amplitude/1000
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


    def create_job(self, sequence, index, seq_id, n_rep, sample_rate, neutralize=True):
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

        aggregator = UploadAggregator(self.awg_channels, self.marker_channels, self.qubit_channels)

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

        # queue waveforms
        for channel_name, queue in job.channel_queues.items():
            offset = 0
            amplitude = AwgConfig.MAX_AMPLITUDE/1000

            if channel_name in self.awg_channels:
                channel = self.awg_channels[channel_name]
                awg_name = channel.awg_name
                channel_number = channel.channel_number
            elif channel_name in self.marker_channels:
                channel = self.marker_channels[channel_name]
                awg_name = channel.module_name
                channel_number = channel.channel_number
                if channel.invert:
                    offset = channel.amplitude/1000
                    amplitude = -amplitude
            else:
                raise Exception(f'Undefined channel {channel_name}')

            self.AWGs[awg_name].set_channel_amplitude(amplitude, channel_number)
            self.AWGs[awg_name].set_channel_offset(offset, channel_number)

            # empty AWG queue
            self.AWGs[awg_name].awg_flush(channel_number)

            start_delay = 0 # no start delay
            trigger_mode = 1 # software/HVI trigger
            cycles = 1
            for queue_item in queue:
                awg = self.AWGs[awg_name]
                prescaler = awg.convert_sample_rate_to_prescaler(queue_item.sample_rate)
                awg.awg_queue_waveform(
                        channel_number, queue_item.wave_reference,
                        trigger_mode, start_delay, cycles, prescaler)
                trigger_mode = 0 # Auto tigger -- next waveform will play automatically.

        # start hvi (start function loads schedule if not yet loaded)
        job.hw_schedule.set_configuration(job.schedule_params, job.n_waveforms)
        job.hw_schedule.start(job.playback_time, job.n_rep, job.schedule_params)

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
    attenuation: float = 1.0
    dc_compensation: bool = False
    dc_compensation_min: float = 0.0
    dc_compensation_max: float = 0.0
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

    def __init__(self, awg_channels, marker_channels, qubit_channels):
        self.npt = 0
        self.marker_channels = marker_channels
        self.qubit_channels = qubit_channels
        self.channels = dict()

        delays = []
        for channel in awg_channels.values():
            info = ChannelInfo()
            self.channels[channel.name] = info

            info.attenuation = channel.attenuation
            info.delay_ns = channel.delay
            delays.append(channel.delay)

            # Note: Compensation limits are specified before attenuation, i.e. at AWG output level.
            #       Convert compensation limit to device level.
            info.dc_compensation_min = channel.compensation_limits[0] * info.attenuation
            info.dc_compensation_max = channel.compensation_limits[1] * info.attenuation
            info.dc_compensation = info.dc_compensation_min < 0 and info.dc_compensation_max > 0

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
                    seg_ch = getattr(seg, channel_name)
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
            duration = seg.total_time[tuple(job.index)]
            npt =  int(round(duration * sample_rate))
            info = SegmentRenderInfo(sample_rate, t_start, npt)
            segments.append(info)
            t_start = info.t_end

        # sections
        sections = job.upload_info.sections
        t_start = -max_pre_start_ns
        nseg = len(segments)

        section = RenderSection(segments[0].sample_rate, t_start)
        sections.append(section)
        section.npt += round(max_pre_start_ns * section.sample_rate)

        for iseg,seg in enumerate(segments):
            sample_rate = seg.sample_rate

            if iseg < nseg-1:
                sample_rate_next = segments[iseg+1].sample_rate
            else:
                sample_rate_next = 0

            # create welding region if sample_rate decreases
            if sample_rate < section.sample_rate:
                # welding region is length of padding for alignment + post_stop region
                n_post = round(((seg.t_start + max_post_end_ns) - section.t_end) * section.sample_rate)
                section.npt += n_post
                section.align(extend=True)

                # number of points of segment to be rendered to previous section
                n_start_transition = round((section.t_end - seg.t_start)*sample_rate)

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
                n_pre = round((section.t_end - (seg.t_end - max_pre_start_ns)) * section.sample_rate)
                section.npt -= n_pre
                section.align(extend=True)

                # start new section
                section = RenderSection(sample_rate_next, section.t_end)
                sections.append(section)

                # number of points of segment to be rendered to next section
                n_end_transition = round((section.t_start - seg.t_end)*sample_rate_next)

                section.npt += n_end_transition

                seg.n_end_transition = n_end_transition
                seg.end_section = section

        # add post stop samples; seg = last segment, section is last section
        n_post = round((section.t_end - (seg.t_end + max_post_end_ns)) * section.sample_rate)
        section.npt += n_post

        # add DC compensation
        compensation_time = self.get_max_compensation_time()
        logging.debug(f'DC compensation time: {compensation_time*1e9} ns')
        compensation_npt = int(np.ceil(compensation_time * section.sample_rate * 1e9))

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


    def _generate_upload(self, job, awg_upload_func):
        segments = self.segments
        sections = job.upload_info.sections
        ref_channel_states = RefChannels(0)

        # loop over all qubit channels to accumulate total phase shift
        for i in range(len(job.sequence)):
            ref_channel_states.start_phases_all.append(dict())
        for channel_name, qubit_channel in self.qubit_channels.items():
            phase = 0
            for iseg,seg in enumerate(job.sequence):
                ref_channel_states.start_phases_all[iseg][channel_name] = phase
                print(f'phase: {channel_name}.{iseg}: {phase}')
                seg_ch = getattr(seg, channel_name)
                phase += seg_ch.get_accumulated_phase(job.index)

        for channel_name, channel_info in self.channels.items():
            section = sections[0]
            buffer = np.zeros(section.npt)

            for iseg,(seg,seg_render) in enumerate(zip(job.sequence,segments)):

                sample_rate = seg_render.sample_rate
                n_delay = round(channel_info.delay_ns * sample_rate)

                seg_ch = getattr(seg, channel_name)
                ref_channel_states.start_time = seg_render.t_start
                ref_channel_states.start_phase = ref_channel_states.start_phases_all[iseg]
                start = time.perf_counter()
                print(f'start: {channel_name}.{iseg}: {ref_channel_states.start_time}')
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
#                    n_delay_welding = round(channel_info.delay_ns * section.sample_rate)
                    t_welding = (section.t_end - seg_render.t_start)
                    i_start = round(t_welding*sample_rate) - n_delay
                    n_section = round(t_welding*section.sample_rate) + round(-channel_info.delay_ns * section.sample_rate)

                    if n_section > 0:
                        if np.round(n_section*sample_rate/section.sample_rate) >= len(wvf):
                            raise Exception(f'segment {iseg} too short for welding. (nwelding:{n_section}, len_wvf:{len(wvf)})')

                        isub = [np.round(i*sample_rate/section.sample_rate) for i in np.arange(n_section)]
                        welding_samples = np.take(wvf, isub)
                        buffer[-n_section:] = welding_samples

                    self._upload_wvf(job, channel_name, channel_info.attenuation, buffer, section.sample_rate, awg_upload_func)

                    section = seg_render.section
                    buffer = np.zeros(section.npt)


                if seg_render.end_section:
                    next_section = seg_render.end_section
                    # add n_end_transition + n_delay to next section. First complete this section
                    n_delay_welding = round(channel_info.delay_ns * section.sample_rate)
                    t_welding = (seg_render.t_end - next_section.t_start)
                    i_end = len(wvf) - round(t_welding*sample_rate) + n_delay_welding

                    if i_start != i_end:
                        buffer[-(i_end-i_start):] = wvf[i_start:i_end]

                    self._upload_wvf(job, channel_name, channel_info.attenuation, buffer, section.sample_rate, awg_upload_func)

                    section = next_section
                    buffer = np.zeros(section.npt)

                    n_section = round(t_welding*section.sample_rate) + round(channel_info.delay_ns * section.sample_rate)
                    if np.round(n_section*sample_rate/section.sample_rate) >= len(wvf):
                        raise Exception(f'segment {iseg} too short for welding. (nwelding:{n_section}, len_wvf:{len(wvf)})')

                    isub = [min(len(wvf)-1, i_end + np.round(i*sample_rate/section.sample_rate)) for i in np.arange(n_section)]
                    welding_samples = np.take(wvf, isub)
                    buffer[:n_section] = welding_samples

                else:
                    if section != seg_render.section:
                        logging.error(f'OOPS-2 section mismatch {iseg}, {channel_name}')
                    offset = seg_render.offset + n_delay
                    buffer[offset+i_start:offset + len(wvf)] = wvf[i_start:]


            if job.neutralize:
                if section != sections[-1]:
                    # Corner case, DC compensation is in a new section # @@@ can this occur??
                    self._upload_wvf(job, channel_name, channel_info.attenuation, buffer, section.sample_rate, awg_upload_func)
                    section = sections[-1]
                    buffer = np.zeros(section.npt)
                    logging.info(f'DC compensation: Corner case {section}')

                compensation_npt = round(job.upload_info.dc_compensation_duration * section.sample_rate)

                if compensation_npt > 0 and channel_info.dc_compensation:
                    compensation_voltage = -channel_info.integral * sample_rate / compensation_npt * 1e9
                    job.upload_info.dc_compensation_voltages[channel_name] = compensation_voltage
                    buffer[-(compensation_npt+1):-1] = compensation_voltage
                    logging.debug(f'DC compensation {channel_name}: {compensation_voltage:6.1f} mV {compensation_npt} Sa')
                else:
                    job.upload_info.dc_compensation_voltages[channel_name] = 0
                    # TODO: @@@ reduce length of waveform?

            self._upload_wvf(job, channel_name, channel_info.attenuation, buffer, section.sample_rate, awg_upload_func)

    def _render_markers(self, job, awg_upload_func):
        for channel_name, marker_channel in self.marker_channels.items():
            logging.debug(f'Marker: {channel_name} ({marker_channel.amplitude} mV)')
            start_stop = []
            for iseg, (seg, seg_render) in enumerate(zip(job.sequence, self.segments)):

                seg_ch = getattr(seg, channel_name)

                ch_data = seg_ch._get_data_all_at(job.index)

                for pulse in ch_data.my_marker_data:
                    start_stop.append((seg_render.t_start + pulse.start - marker_channel.setup_ns, +1))
                    start_stop.append((seg_render.t_start + pulse.stop + marker_channel.hold_ns, -1))

            if len(start_stop) > 0:
                m = np.array(start_stop)
                ind = np.argsort(m[:,0])
                m = m[ind]
            else:
                m = []

            if marker_channel.channel_number == 0:
                self._upload_fpga_markers(job, marker_channel, m)
            else:
                self._upload_awg_markers(job, marker_channel, m, awg_upload_func)

    def _upload_awg_markers(self, job, marker_channel, m, awg_upload_func):
        # TODO @@@ : round section length to 100 ns and use max 1e8 Sa/s rendering?
        sections = job.upload_info.sections
        amplitude = marker_channel.amplitude
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
                    logging.info(f'Warning: Marker setup before waveform; aligned with start')
                    pt_on = 0
                if t_off < section.t_end:
                    pt_off = int((t_off - section.t_start) * section.sample_rate)
                    buffers[i_section][pt_on:pt_off] = amplitude
                else:
                    buffers[i_section][pt_on:] = amplitude
                    i_section += 1
                    # search end section
                    while t_off >= sections[i_section].t_end:
                        buffers[i_section][:] = amplitude
                        i_section += 1
                    section = sections[i_section]
                    pt_off = int((t_off - section.t_start) * section.sample_rate)
                    buffers[i_section][:pt_off] = amplitude

        for buffer, section in zip(buffers, sections):
            self._upload_wvf(job, marker_channel.name, 1.0, buffer, section.sample_rate, awg_upload_func)

    def _upload_fpga_markers(self, job, marker_channel, m):
        table = []
        job.marker_tables[marker_channel.name] = table
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
                table.append((t_on, t_off))

    def _upload_wvf(self, job, channel_name, attenuation, waveform, sample_rate, awg_upload_func):
        # note: numpy inplace multiplication is much faster than standard multiplication
        waveform *= 1/(attenuation * AwgConfig.MAX_AMPLITUDE)
        wave_ref = awg_upload_func(channel_name, waveform)
        job.add_waveform(channel_name, wave_ref, sample_rate*1e9)


    def upload_job(self, job, awg_upload_func):

        job.upload_info = JobUploadInfo()
        job.marker_tables = {} # @@@ add to upload_info?

        self._integrate(job)

        self._generate_sections(job)

        self._generate_upload(job, awg_upload_func)

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


