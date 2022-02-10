import time
import numpy as np
import logging
import copy
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from concurrent.futures.thread import ThreadPoolExecutor

from pulse_lib.segments.data_classes.data_markers import marker_pulse

from .wrapped_5014 import Wrapped5014

class AwgConfig:
    DEFAULT_AMPLITUDE = 1500 # mV

def Tektronix_sync_latency(sample_rate:float):
    '''
    Calculates the sync latency for a Tektronix slave AWG triggered with marker of master Tektronix AWG.
    '''
    decimation = 1.1999e9 / sample_rate
    period_ns = 1e9 / sample_rate
    n = int(np.log2(decimation))
    latency = 22//period_ns*period_ns + period_ns * int(36 + 420 * 2**-n)
    logging.debug(f'Tektronix sync latency {sample_rate/1e6:5.1f} MHz: {latency:5.1f} ns')
    return latency


class Tektronix5014_Uploader:

    verbose = False

    def __init__(self, awgs, awg_channels, marker_channels, digitizer_markers,
                 qubit_channels, digitizer_channels, awg_sync):
        '''
        Initialize the Tektronix uploader.
        Args:
            awgs (dict<awg_name,QcodesIntrument>): list with AWG's
            awg_channels Dict[name, awg_channel]: channel names and properties
            marker_channels: Dict[name, marker_channel]: dict with names and properties
            digitizer_markers: Dict[digitizer, marker_channel]: dict with digitizer names and linked marker
            digitizer_channels: Dict[name, digitizer_channel]: dict with names and properties
            awg_sync: Dict[name, awg_slave]: properties for slave AWGs: marker and latency.
        Returns:
            None
        '''
        self.awgs = awgs

        self.awg_channels = awg_channels
        self.marker_channels = marker_channels
        self.digitizer_channels = digitizer_channels
        self.digitizer_markers = digitizer_markers
        self.qubit_channels = qubit_channels
        self.awg_sync = awg_sync

        self.job = None

        self.setup_slaves()

    def setup_slaves(self):
        logging.info(f'Configure slave AWGs: {self.awg_sync}')
        for slave in self.awg_sync.values():
            awg = self.awgs[slave.awg_name]
            awg.trigger_source('EXT')
            awg.trigger_impedance(1000)
            awg.trigger_level(1.6)
            awg.trigger_slope('POS')

    def get_effective_sample_rate(self, sample_rate):
        """
        Returns the sample rate that will be used by the Tektronix AWG.
        This is the a rate >= requested sample rate.
        """
        return (sample_rate+99)//100 * 100


    def create_job(self, sequence, index, seq_id, n_rep, sample_rate, neutralize=True):
        return Job(sequence, index, seq_id, n_rep, sample_rate, neutralize)


    def add_upload_job(self, job):
        '''
        add a job to the uploader.
        Args:
            job (upload_job) : upload_job object that defines what needs to be uploaded and possible post processing of the waveforms (if needed)
        '''
        start = time.perf_counter()

        job.hw_schedule.stop()

        # calculate sync latency if not set
        awg_sync = copy.deepcopy(self.awg_sync)
        for slave in awg_sync.values():
            if slave.sync_latency is None:
                slave.sync_latency = Tektronix_sync_latency(job.default_sample_rate)

        aggregator = UploadAggregator(self.awgs, self.awg_channels, self.marker_channels,
                                      self.digitizer_channels, self.digitizer_markers,
                                      self.qubit_channels, awg_sync)

        aggregator.upload_job(job)

        self.job = job

        duration = time.perf_counter() - start
        logging.debug(f'generated upload data ({duration*1000:6.3f} ms)')


    def __get_job(self, seq_id, index):
        """
        get job data of an uploaded segment
        Args:
            seq_id (uuid) : id of the sequence
            index (tuple) : index that has to be played
        Return:
            job (upload_job) :job, with locations of the sequences to be uploaded.
        """
        if self.job is not None:
            job = self.job
            if job.seq_id == seq_id and job.index == index and not job.released:
                return job

        logging.error(f'Job not found for index {index} of seq {seq_id}')
        raise ValueError(f'Sequence with id {seq_id}, index {index} not placed for upload .. . Always make sure to first upload your segment and then do the playback.')


    def play(self, seq_id, index, release_job=True):
        """
        start playback of a sequence that has been uploaded.
        Args:
            seq_id (uuid) : id of the sequence
            index (tuple) : index that has to be played
            release_job (bool) : release memory on AWG after done.
        """

        job =  self.__get_job(seq_id, index)
        enable_channels = {awg.name:set() for awg in self.awgs.values()}
        for channel in self.awg_channels.values():
            enable_channels[channel.awg_name].add(channel.channel_number)
        for channel in self.marker_channels.values():
            channel_number = channel.channel_number if isinstance(channel.channel_number, int) else channel.channel_number[0]
            enable_channels[channel.module_name].add(channel_number)

        for awg in self.awgs.values():
            for channel in enable_channels[awg.name]:
                awg.set(f'ch{channel}_state', 1)

        job.hw_schedule.set_configuration(job.schedule_params, job.n_waveforms)
        job.hw_schedule.start(job.playback_time, job.n_rep, job.schedule_params)

    def wait_until_AWG_idle(self):
        while (True):
            not_running = [awg.get_state() != 'Running' for awg in self.awgs.values()]
            if all(not_running):
                break
            time.sleep(0.001)


    def release_memory(self, seq_id, index=None):
        pass


class Job(object):
    """docstring for upload_job"""
    def __init__(self, sequence, index, seq_id, n_rep, sample_rate, neutralize=True, priority=0):
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
        self.sequence = sequence
        self.seq_id = seq_id
        self.index = index
        self.n_rep = n_rep
        self.default_sample_rate = sample_rate
        self.neutralize = neutralize
        self.priority = priority
        self.playback_time = 0 #total playtime of the waveform

        self.released = False

        self.hw_schedule = None
        self.digitizer_triggers = []
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
        pass

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

    def __init__(self, awgs, awg_channels, marker_channels, digitizer_channels,
                 digitizer_markers, qubit_channels, awg_sync):
        self.npt = 0
        self.marker_channels = marker_channels
        self.digitizer_channels = digitizer_channels
        self.digitizer_markers = digitizer_markers
        self.qubit_channels = qubit_channels
        self.awg_sync = awg_sync
        self.channels = dict()
        self.waveforms = dict()
        self.awgs = dict()
        for awg in awgs.values():
            self.awgs[awg.name] = Wrapped5014(awg, awg_channels, marker_channels)

        delays = []
        for channel in awg_channels.values():
            info = ChannelInfo()
            self.channels[channel.name] = info

            slave = self.awg_sync.get(channel.awg_name, None)
            sync_latency = 0 if slave is None else slave.sync_latency

            info.amplitude = channel.amplitude if channel.amplitude is not None else AwgConfig.DEFAULT_AMPLITUDE
            info.attenuation = channel.attenuation
            info.bias_T_RC_time = channel.bias_T_RC_time
            info.delay_ns = channel.delay - sync_latency
            delays.append(channel.delay - sync_latency)

            # Note: Compensation limits are specified before attenuation, i.e. at AWG output level.
            #       Convert compensation limit to device level.
            info.dc_compensation_min = channel.compensation_limits[0] * info.attenuation
            info.dc_compensation_max = channel.compensation_limits[1] * info.attenuation
            info.dc_compensation = info.dc_compensation_min < 0 and info.dc_compensation_max > 0

        slave_markers = {slave.marker_name:slave for slave in self.awg_sync.values()}
        for channel in marker_channels.values():
            slave = slave_markers.get(channel.name, None)
            sync_latency = 0 if slave is None else slave.sync_latency
            delays.append(channel.delay - sync_latency - channel.setup_ns)
            delays.append(channel.delay - sync_latency + channel.hold_ns)

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
        sample_rate = job.default_sample_rate * 1e-9

        self.segments = []
        segments = self.segments
        t_start = 0
        for seg in job.sequence:
            # work with sample rate in GSa/s
            if seg.sample_rate and seg.sample_rate != job.default_sample_rate:
                raise Exception('multipe sample rates is not supported for Tektronix')
            duration = seg.get_total_time(job.index)
            npt =  int(duration * sample_rate + 0.5)
            info = SegmentRenderInfo(sample_rate, t_start, npt)
            segments.append(info)
            t_start = info.t_end

        # sections
        sections = job.upload_info.sections
        # add at least 2 zero in front, because Tek outputs first sample when waiting for start trigger.
        # add the second to compensate for rounding errors in rendering.
        start_samples = 2
        t_start = -max_pre_start_ns - start_samples/segments[0].sample_rate

        section = RenderSection(segments[0].sample_rate, t_start)
        sections.append(section)
        section.npt += start_samples
        section.npt += int(max_pre_start_ns * section.sample_rate + 0.5)

        for iseg,seg in enumerate(segments):
            seg.section = section
            seg.offset = section.npt
            section.npt += seg.npt

        # add post stop samples; seg = last segment, section is last section
        n_post = int(((seg.t_end + max_post_end_ns) - section.t_end) * section.sample_rate + 0.5)
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


    def _generate_upload(self, job):
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
                #print(f'phase: {channel_name}.{iseg}: {phase}')
                seg_ch = getattr(seg, channel_name)
                phase += seg_ch.get_accumulated_phase(job.index)

        for channel_name, channel_info in self.channels.items():
            section = sections[0]
            buffer = np.zeros(section.npt)
            bias_T_compensation_mV = 0

            for iseg,(seg,seg_render) in enumerate(zip(job.sequence,segments)):

                sample_rate = seg_render.sample_rate
                n_delay = int(channel_info.delay_ns * sample_rate + 0.5)

                seg_ch = getattr(seg, channel_name)
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
                if section != seg_render.section:
                    logging.error(f'OOPS-2 section mismatch {iseg}, {channel_name}')
                offset = seg_render.offset + n_delay
                buffer[offset+i_start:offset + len(wvf)] = wvf[i_start:]


            if job.neutralize:
                compensation_npt = round(job.upload_info.dc_compensation_duration * section.sample_rate)

                if compensation_npt > 0 and channel_info.dc_compensation:
                    compensation_voltage = -channel_info.integral * sample_rate / compensation_npt * 1e9
                    job.upload_info.dc_compensation_voltages[channel_name] = compensation_voltage
                    buffer[-(compensation_npt+1):-1] = compensation_voltage
                    logging.debug(f'DC compensation {channel_name}: {compensation_voltage:6.1f} mV {compensation_npt} Sa')
                else:
                    job.upload_info.dc_compensation_voltages[channel_name] = 0

            bias_T_compensation_mV = self._add_bias_T_compensation(buffer, bias_T_compensation_mV,
                                                                   section.sample_rate, channel_info)
            self._upload_wvf(job, channel_name, buffer, channel_info.attenuation, channel_info.amplitude)

    def _generate_digitizer_triggers(self, job):
        triggers = set()

        for name, value in job.schedule_params.items():
            if name.startswith('dig_trigger_'):
                triggers.add(value)

        for channel_name, channel in self.digitizer_channels.items():
            for iseg, (seg, seg_render) in enumerate(zip(job.sequence, self.segments)):
                seg_ch = getattr(seg, channel_name)
                acquisition_data = seg_ch._get_data_all_at(job.index).get_data()
                for acquisition in acquisition_data:
                    triggers.add(seg_render.t_start + acquisition.start)

        job.digitizer_triggers = list(triggers)
        job.digitizer_triggers.sort()
        logging.info(f'digitizer triggers: {job.digitizer_triggers}')

    def _generate_digitizer_markers(self, job):
        pulse_duration = max(100, 1e9/job.default_sample_rate) # 1 Sample or 100 ns
        marker_data = []
        for t in job.digitizer_triggers:
                marker_data.append(marker_pulse(t, t + pulse_duration))
        return marker_data

    def _render_markers(self, job):
        slave_markers = {slave.marker_name:slave for slave in self.awg_sync.values()}

        for channel_name, marker_channel in self.marker_channels.items():
            logging.debug(f'Marker: {channel_name} ({marker_channel.amplitude} mV, {marker_channel.delay:+2.0f} ns)')
            start_stop = []
            for iseg, (seg, seg_render) in enumerate(zip(job.sequence, self.segments)):

                seg_ch = getattr(seg, channel_name)

                ch_data = seg_ch._get_data_all_at(job.index)

                for pulse in ch_data.my_marker_data:
                    offset = seg_render.t_start + marker_channel.delay
                    start_stop.append((offset + pulse.start - marker_channel.setup_ns, +1))
                    start_stop.append((offset + pulse.stop + marker_channel.hold_ns, -1))

            if channel_name in self.digitizer_markers.values():
                pulses = self._generate_digitizer_markers(job)
                logging.info(f'dig trigger: {pulses}')
                for pulse in pulses:
                    # trigger time is relative to sequence start, not segment start
                    offset = marker_channel.delay
                    start_stop.append((offset + pulse.start - marker_channel.setup_ns, +1))
                    start_stop.append((offset + pulse.stop + marker_channel.hold_ns, -1))


            if channel_name in slave_markers:
                slave = slave_markers[channel_name]
                offset = marker_channel.delay - slave.sync_latency
                # trigger time is relative to sequence start, not segment start
                start_stop.append((offset - marker_channel.setup_ns, +1))
                start_stop.append((offset + marker_channel.hold_ns, -1))

            if len(start_stop) > 0:
                m = np.array(start_stop)
                ind = np.argsort(m[:,0])
                m = m[ind]
            else:
                m = []

            self._upload_awg_markers(job, marker_channel, m)

    def _upload_awg_markers(self, job, marker_channel, m):
        sections = job.upload_info.sections
        section = sections[0]
        buffer = np.zeros(section.npt, dtype=np.float32)
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
                logging.debug(f'Marker: {t_on} - {t_off} ({section.t_start:+}ns)')
                # search start section
                if t_on >= section.t_end:
                    logging.error(f'Failed to render marker t_on > start')
                pt_on = int((t_on - section.t_start) * section.sample_rate)
                if pt_on < 0:
                    logging.info(f'Warning: Marker setup before waveform; aligned with start')
                    pt_on = 0
                if t_off < section.t_end:
                    pt_off = int((t_off - section.t_start) * section.sample_rate)
                    buffer[pt_on:pt_off] = 1.0
                else:
                    logging.error(f'Failed to render marker t_off > end')

        self._upload_wvf(job, marker_channel.name, buffer, 1.0, 1.0)


    def _upload_wvf(self, job, channel_name, waveform, attenuation, amplitude):
        # note: numpy inplace multiplication is much faster than standard multiplication
        waveform *= 1/(attenuation * amplitude)
        self.waveforms[channel_name] = waveform


    def upload_job(self, job):

        job.upload_info = JobUploadInfo()

        self._integrate(job)

        self._generate_sections(job)

        self._generate_upload(job)

        self._generate_digitizer_triggers(job)

        self._render_markers(job)

        self.upload(job)


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


    def _upload_waveforms(self, job, awg):
            awg.set_sample_rate(job.default_sample_rate)
            # awg filters waveforms on name
            awg.upload_waveforms(self.waveforms)

    def upload(self, job):
        use_concurrent_upload = True
        if use_concurrent_upload:
            with ThreadPoolExecutor() as p:
                result = p.map(
                    lambda awg: self._upload_waveforms(job, awg),
                    self.awgs.values()
                    )
            for r in result:
                # loop through the results to raise exceptions caught by Executor
                if result is not None:
                    logging.debug(f'upload result {r}')
        else:
            for awg in self.awgs.values():
                self._upload_waveforms(job, awg)



