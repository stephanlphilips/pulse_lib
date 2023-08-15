import time
import logging
import copy
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from uuid import UUID
from concurrent.futures.thread import ThreadPoolExecutor

import numpy as np

from pulse_lib.segments.data_classes.data_markers import marker_pulse
from pulse_lib.uploader.uploader_funcs import merge_markers
from pulse_lib.uploader.digitizer_triggers import DigitizerTriggerBuilder, DigitizerTriggers

try:
    from .m4i_controller import M4iControl
except:
    def M4iControl(*args, **kwargs):
        raise Exception('Import of M4iControl failed')


logger = logging.getLogger(__name__)


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
    logger.debug(f'Tektronix sync latency {sample_rate/1e6:5.1f} MHz: {latency:5.1f} ns')
    return latency


class Tektronix5014_Uploader:

    verbose = False

    def __init__(self, awgs, digitizers,
                 awg_channels, marker_channels, digitizer_markers,
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
        self.digitizers = digitizers

        self.awg_channels = awg_channels
        self.marker_channels = marker_channels
        self.digitizer_channels = digitizer_channels
        self.digitizer_markers = digitizer_markers
        self.qubit_channels = qubit_channels
        self.awg_sync = awg_sync

        self.jobs = []
        self.last_job = None

        self.setup_slaves()
        self.set_cfg()

        self.pending_deletes = dict()
        self.release_all_awg_memory()

    def setup_slaves(self):
        logger.info(f'Configure slave AWGs: {self.awg_sync}')
        for slave in self.awg_sync.values():
            awg = self.awgs[slave.awg_name]
            awg.trigger_source('EXT')
            awg.trigger_impedance(1000)
            awg.trigger_level(1.6)
            awg.trigger_slope('POS')

    def set_cfg(self): # @@@ update also when changed.
        max_amplitude = 4500 / 2
        min_amplitude = 20 / 2
        for channel in self.awg_channels.values():
            amplitude = channel.amplitude if channel.amplitude is not None else AwgConfig.DEFAULT_AMPLITUDE
            awg = self.awgs[channel.awg_name]
            if amplitude > max_amplitude or amplitude < min_amplitude:
                raise ValueError(f'amplitude ({amplitude}) out of range [{min_amplitude}, {max_amplitude}] mV')
            ch_num = channel.channel_number
            # NOTE: Tektronix setting is Vpp, not amplitude.
            awg.set(f'ch{ch_num}_amp', amplitude/1000*2)
            offset = channel.offset if channel.offset is not None else 0
            awg.set(f'ch{ch_num}_offset', offset/1000)

        for channel in self.marker_channels.values():
            awg = self.awgs[channel.module_name]
            amplitude = channel.amplitude if channel.amplitude is not None else AwgConfig.DEFAULT_AMPLITUDE
            if isinstance(channel.channel_number, (tuple, list)):
                if amplitude > 2700 or amplitude < -900:
                    raise ValueError(f'marker amplitude ({amplitude}) out of range [-900, 2700] mV')
                channel_number = channel.channel_number[0]
                marker_number = channel.channel_number[1]
                if channel.invert: # @@@ doesn't work?
                    awg.set(f'ch{channel_number}_m{marker_number}_low', amplitude/1000)
                    awg.set(f'ch{channel_number}_m{marker_number}_high', 0.0)
                else:
                    awg.set(f'ch{channel_number}_m{marker_number}_high', amplitude/1000)
                    awg.set(f'ch{channel_number}_m{marker_number}_low', 0.0)
            else:
                if amplitude > max_amplitude or amplitude < min_amplitude:
                    raise ValueError(f'amplitude ({amplitude}) out of range [{min_amplitude}, {max_amplitude}] mV')
                ch_num = channel.channel_number
                awg.set(f'ch{ch_num}_amp', amplitude/1000*2)

    def get_effective_sample_rate(self, sample_rate):
        """
        Returns the sample rate that will be used by the Tektronix AWG.
        This is the a rate >= requested sample rate.
        """
        return (sample_rate+99)//100 * 100


    def create_job(self, sequence, index, seq_id, n_rep, sample_rate, neutralize=True, alignment=None):
        self.release_memory(seq_id, index)
        return Job(self.jobs, sequence, index, seq_id, n_rep, sample_rate, neutralize,
                   self.pending_deletes, alignment=alignment)


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

        self.jobs.append(job)

        duration = time.perf_counter() - start
        logger.debug(f'generated upload data ({duration*1000:6.3f} ms)')


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

        # load sequence if needed
        if job != self.last_job:
            if self.last_job is not None:
                self.last_job.hw_schedule.stop()
            self._set_sequence(job)
            self.last_job = job

        self._delete_released_waveforms()

        self._configure_digitizers(job)
        job.hw_schedule.set_configuration(job.schedule_params, job.n_waveforms)
        n_rep = job.n_rep if job.n_rep else 1
        job.hw_schedule.start(job.playback_time, n_rep, job.schedule_params)

        if release_job:
            job.release()

    def _delete_released_waveforms(self):
        # remove old waveforms
        for awg in self.awgs.values():
            for waveform_name in self.pending_deletes[awg.name]:
                _delete_waveform(awg, waveform_name)
            self.pending_deletes[awg.name] = []

    def wait_until_AWG_idle(self):
        while (True):
            not_running = [awg.get_state() != 'Running' for awg in self.awgs.values()]
            if all(not_running):
                break
            time.sleep(0.001)

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
        self._delete_released_waveforms()

    def release_all_awg_memory(self):
        for awg in self.awgs.values():
            awg.sequence_length(0)
            awg.delete_all_waveforms_from_list()
            self.pending_deletes[awg.name] = []
        self.release_memory()

    def _set_sequence(self, job):
        for awg in self.awgs.values():
            awg_data = job.waveform_data[awg.name]
            n_elements = 1
            awg.sequence_length(n_elements)
            element_no = 1
            enable_channels = []
            for channel in range(1,5):
                if channel in awg_data:
                    wave_name = awg_data[channel].name
                    awg.set_sqel_waveform(wave_name, channel, element_no)
                    enable_channels.append(channel)
                else:
                    awg.set_sqel_waveform("", channel, element_no)
            awg.set_sqel_goto_state(element_no, 1)

            for channel in enable_channels:
                awg.set(f'ch{channel}_state', 1)

    def _configure_digitizers(self, job):
        if not job.acquisition_conf.configure_digitizer:
            return
        '''
        Configure per digitizer channel:
            n_triggers: job.n_triggers per channel @@@ all equal
            t_measure: job.t_measure per channel @@@ all equal
            downsampled_rate:
        '''
        '''
        Read:
            * reshape for n_rep, n_trigger
            * aggregate: average of t_measure / down-sample
            * filter data after, remove unused values.
            *
        '''
        acq_conf = job.acquisition_conf

        # Currently support 1 digitizer
        digitizer = list(self.digitizers.values())[0]
        self.m4i_control = M4iControl(digitizer)
        self.m4i_control.configure_acquisitions(
                job.digitizer_triggers,
                job.n_rep,
                average_repetitions=acq_conf.average_repetitions)

        self.acq_description = AcqDescription(job.seq_id, job.index,
                                              job.digitizer_triggers)

    def get_channel_data(self, seq_id, index):
        acq_desc = self.acq_description
        if (acq_desc.seq_id != seq_id
            or (index is not None and acq_desc.index != index)):
            raise Exception(f'Data for index {index} not available')

        dig_data = self.m4i_control.get_data()
        data = {i:np.zeros(0) for i in [1,2,3,4]}
        for i,ch in enumerate(acq_desc.digitizer_triggers.active_channels):
            data[ch] = dig_data[i]

        result = {}
        for channel_name,channel in self.digitizer_channels.items():
            in_ch = channel.channel_numbers
            if len(in_ch) == 2:
                raw_I = data[in_ch[0]]
                raw_Q = data[in_ch[1]]
                raw_ch = (raw_I + 1j * raw_Q) * np.exp(1j*channel.phase)
            else:
                raw_ch = data[in_ch[0]]

            if not channel.iq_out:
                raw_ch = raw_ch.real

            result[channel_name] = raw_ch

        return result

@dataclass
class ChannelData:
    name: str
    length: int
    wvf: Optional[np.ndarray] = None
    m1: Optional[np.ndarray] = None
    m2: Optional[np.ndarray] = None

class Job(object):
    last_job_id = 0

    """docstring for upload_job"""
    def __init__(self, job_list, sequence, index, seq_id, n_rep, sample_rate, neutralize,
                 delete_list, alignment=None):
        '''
        Args:
            job_list (list): list with all jobs.
            sequence (list of list): list with list of the sequence
            index (tuple) : index that needs to be uploaded
            seq_id (uuid) : if of the sequence
            n_rep (int) : number of repetitions of this sequence.
            sample_rate (float) : sample rate
            neutralize (bool) : place a neutralizing segment at the end of the upload
            delete_list (dict[str, list]): list per AWG for adding waveforms to be deleted.
            alignment (optional int): repetition period alignment in ns.
        '''
        self.job_list = job_list
        self.sequence = sequence
        self.seq_id = seq_id
        self.index = index
        self.n_rep = n_rep
        self.default_sample_rate = sample_rate
        self.neutralize = neutralize
        self.delete_list = delete_list
        self.alignment = alignment
        self.playback_time = 0 #total playtime of the waveform
        self.acquisition_conf = None

        self.released = False
        self.job_id = self._get_job_id()

        self.hw_schedule = None
        self.digitizer_triggers = None
        # waveform data per awg and channel: Dict[str,Dict[int,ChannelData]]
        self.waveform_data = {}
        logger.debug(f'new job {self.job_id:04X} ({seq_id}-{index})')

    def _get_job_id(self):
        all_ids = [job.job_id for job in self.job_list]
        good_id = False
        while not good_id:
            Job.last_job_id = (Job.last_job_id + 1) % 0x10000
            good_id = Job.last_job_id not in all_ids
        return Job.last_job_id

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

    def release(self):
        if self.released:
            logger.warning(f'job {self.job_id:04X} ({self.seq_id}-{self.index}) already released')
            return

        logger.debug(f'release job {self.seq_id}-{self.index}')
        self.released = True

        for awg_name, awg_data in self.waveform_data.items():
            for channel_data in awg_data.values():
                self.delete_list[awg_name].append(channel_data.name)

        if self in self.job_list:
            self.job_list.remove(self)


    def __del__(self):
        if not self.released:
            logger.warning(f'Job {self.job_id} ({self.seq_id}-{self.index}) was not released. '
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


@dataclass
class RfMarkerPulse:
    start: float
    stop: float


@dataclass
class AcqDescription:
    seq_id: UUID
    index: List[int]
    digitizer_triggers: DigitizerTriggers


class UploadAggregator:
    verbose = False

    def __init__(self, awgs, awg_channels, marker_channels, digitizer_channels,
                 digitizer_markers, qubit_channels, awg_sync):
        self.npt = 0
        self.awgs = awgs
        self.awg_channels = awg_channels
        self.marker_channels = marker_channels
        self.digitizer_channels = digitizer_channels
        self.digitizer_markers = digitizer_markers
        self.qubit_channels = qubit_channels
        self.awg_sync = awg_sync
        self.channels = dict()
        self.waveform_data = dict()
        for awg_name in self.awgs:
            self.waveform_data[awg_name] = dict()

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

        for channel in digitizer_channels.values():
            delays.append(channel.delay)
            if channel.rf_source is not None:
                rf_source = channel.rf_source
                delays.append(rf_source.delay - rf_source.startup_time_ns)
                delays.append(rf_source.delay + rf_source.prolongation_ns)

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
                    logger.debug(f'Integral seg:{iseg} {channel_name} integral:{channel_info.integral}')


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
                raise Exception('multiple sample rates is not supported for Tektronix')
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
        logger.debug(f'DC compensation time: {compensation_time*1e9} ns')
        compensation_npt = int(np.ceil(compensation_time * section.sample_rate * 1e9))

        job.upload_info.dc_compensation_duration = compensation_npt/section.sample_rate
        section.npt += compensation_npt

        # add at least 1 zero
        section.npt += 1
        if job.alignment:
            alignment = job.alignment
            section.npt = (int(section.npt + alignment - 1) // alignment) * alignment

        job.playback_time = section.t_end - sections[0].t_start
        job.n_waveforms = len(sections)
        logger.debug(f'Playback time: {job.playback_time} ns')

        if UploadAggregator.verbose:
            for segment in segments:
                logger.info(f'segment: {segment}')
            for section in sections:
                logger.info(f'section: {section}')


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
                logger.debug(f'generated [{job.index}]{iseg}:{channel_name} {len(wvf)} Sa, in {duration*1000:6.3f} ms')

                if len(wvf) != seg_render.npt:
                    logger.warning(f'waveform {iseg}:{channel_name} {len(wvf)} Sa <> sequence length {seg_render.npt}')

                i_start = 0
                if section != seg_render.section:
                    logger.error(f'OOPS-2 section mismatch {iseg}, {channel_name}')
                offset = seg_render.offset + n_delay
                buffer[offset+i_start:offset + len(wvf)] = wvf[i_start:]


            if job.neutralize:
                compensation_npt = round(job.upload_info.dc_compensation_duration * section.sample_rate)

                if compensation_npt > 0 and channel_info.dc_compensation:
                    compensation_voltage = -channel_info.integral * sample_rate / compensation_npt * 1e9
                    job.upload_info.dc_compensation_voltages[channel_name] = compensation_voltage
                    buffer[-(compensation_npt+1):-1] = compensation_voltage
                    logger.debug(f'DC compensation {channel_name}: {compensation_voltage:6.1f} mV {compensation_npt} Sa')
                else:
                    job.upload_info.dc_compensation_voltages[channel_name] = 0

            bias_T_compensation_mV = self._add_bias_T_compensation(buffer, bias_T_compensation_mV,
                                                                   section.sample_rate, channel_info)
            self._upload_wvf(job, channel_name, buffer, channel_info.attenuation, channel_info.amplitude)

    def _generate_digitizer_triggers(self, job):
        acq_conf = job.acquisition_conf
        digitizer_triggers = DigitizerTriggerBuilder(acq_conf.t_measure, acq_conf.sample_rate)
        self.rf_marker_pulses = {}

        for name, value in job.schedule_params.items():
            if name.startswith('dig_trigger_'):
                if acq_conf.configure_digitizer:
                    raise Exception('{name} cannot be used when digitizer must be configured')
                # channel number will not be used.
                digitizer_triggers.add_acquisition(1, value)

        for channel_name, channel in self.digitizer_channels.items():
            rf_source = channel.rf_source
            if rf_source is not None:
                rf_marker_pulses = []
                self.rf_marker_pulses[rf_source.output] = rf_marker_pulses # @@@ FAILS with multiple channels.


            t_end = None
            for iseg, (seg, seg_render) in enumerate(zip(job.sequence, self.segments)):
                seg_ch = seg[channel_name]
                acquisition_data = seg_ch._get_data_all_at(job.index).get_data()
                for acquisition in acquisition_data:
                    t = seg_render.t_start + acquisition.start
                    digitizer_triggers.add_acquisition(channel.channel_numbers,
                                                       t+channel.delay, acquisition.t_measure)
                    t_measure = acquisition.t_measure if acquisition.t_measure is not None else acq_conf.t_measure
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

        job.digitizer_triggers = digitizer_triggers.get_result()
        if UploadAggregator.verbose:
            logger.debug(f'digitizer triggers: {job.digitizer_triggers.triggered_channels}')

    def _generate_digitizer_markers(self, job):
        pulse_duration = max(100, 1e9/job.default_sample_rate) # 1 Sample or 100 ns
        marker_data = []
        for t in job.digitizer_triggers.triggers:
            marker_data.append(marker_pulse(t, t + pulse_duration))
        return marker_data

    def _render_markers(self, job):
        slave_markers = {slave.marker_name:slave for slave in self.awg_sync.values()}

        for channel_name, marker_channel in self.marker_channels.items():
            logger.debug(f'Marker: {channel_name} ({marker_channel.amplitude} mV, {marker_channel.delay:+2.0f} ns)')
            offset = marker_channel.delay
            start_stop = []
            if channel_name in self.rf_marker_pulses:
                for pulse in self.rf_marker_pulses[channel_name]:
                    start_stop.append((offset + pulse.start - marker_channel.setup_ns, +1))
                    start_stop.append((offset + pulse.stop + marker_channel.hold_ns, -1))

            for iseg, (seg, seg_render) in enumerate(zip(job.sequence, self.segments)):

                seg_ch = getattr(seg, channel_name)

                ch_data = seg_ch._get_data_all_at(job.index)

                for pulse in ch_data.my_marker_data:
                    offset = seg_render.t_start + marker_channel.delay
                    start_stop.append((offset + pulse.start - marker_channel.setup_ns, +1))
                    start_stop.append((offset + pulse.stop + marker_channel.hold_ns, -1))

            if channel_name in self.digitizer_markers.values():
                pulses = self._generate_digitizer_markers(job)
                logger.info(f'dig trigger: {pulses}')
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

            m = merge_markers(channel_name, start_stop)
            self._upload_awg_markers(job, marker_channel, m)

    def _upload_awg_markers(self, job, marker_channel, m):
        sections = job.upload_info.sections
        section = sections[0]
        buffer = np.zeros(section.npt, dtype=np.uint16)
        for i in range(0, len(m), 2):
            t_on = m[i][0]
            t_off = m[i+1][0]
            if UploadAggregator.verbose:
                logger.debug(f'Marker: {t_on} - {t_off} ({section.t_start:+}ns)')
            # search start section
            if t_on >= section.t_end:
                logger.error(f'Failed to render marker t_on > start')
            pt_on = int((t_on - section.t_start) * section.sample_rate)
            if pt_on < 0:
                logger.warning(f'Warning: Marker setup before waveform; aligned with start')
                pt_on = 0
            if t_off > section.t_end:
                logger.warning(f'Truncated marker {marker_channel.name} at {section.t_end}')
                t_off = section.t_end
            pt_off = int((t_off - section.t_start) * section.sample_rate)
            buffer[pt_on:pt_off] = 1

        buffer[-1] = 0
#        self._upload_wvf(job, marker_channel.name, buffer, 1.0, 1.0)
        self._add_channel_data(job, marker_channel.name, buffer)


    def _upload_wvf(self, job, channel_name, waveform, attenuation, amplitude):
        # note: numpy inplace multiplication is much faster than standard multiplication
        waveform *= 1/(attenuation * amplitude)
        self._add_channel_data(job, channel_name, waveform)

    def _add_channel_data(self, job, channel_name, data):
        marker_number = None
        if channel_name in self.awg_channels:
            awg_channel = self.awg_channels[channel_name]
            channel_number = awg_channel.channel_number
            awg_name = awg_channel.awg_name
        elif channel_name in self.marker_channels:
            marker_channel = self.marker_channels[channel_name]
            awg_name = marker_channel.module_name
            channel_number = marker_channel.channel_number
            if isinstance(channel_number, tuple):
                marker_number = channel_number[1]
                channel_number = channel_number[0]
        else:
            raise Exception(f'Unknown channel {channel_name}')

        awg_data = self.waveform_data[awg_name]
        if channel_number not in awg_data:
            l = len(data)
            channel_data = ChannelData(f'ch_{channel_number}_x{job.job_id:04X}', l)
            awg_data[channel_number] = channel_data
        else:
            channel_data = awg_data[channel_number]

        if not marker_number:
            channel_data.wvf = data
        elif marker_number == 1:
            channel_data.m1 = data
        else:
            channel_data.m2 = data


    def upload_job(self, job):

        job.upload_info = JobUploadInfo()
        job.waveform_data = self.waveform_data

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
            logger.info(f'bias-T compensation  min:{np.min(compensation):5.1f} max:{np.max(compensation):5.1f} mV')
            buffer += compensation

        return bias_T_compensation_mV


    def _upload_waveforms(self, job, awg):

        awg.clock_freq(int(job.default_sample_rate))
        for data in job.waveform_data[awg.name].values():
            _send_waveform_to_list(awg, data.length, data.wvf, data.m1, data.m2, data.name)

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
                    logger.debug(f'upload result {r}')
        else:
            for awg in self.awgs.values():
                self._upload_waveforms(job, awg)


def _delete_waveform(awg, name):
    s = f'WLISt:WAVeform:DEL "{name}"'
    awg.write(s)


def _send_waveform_to_list(awg, length, wf, m1, m2, name):
    ''' Send a waveform to the waveform list of the AWG.
    Note: This is a rewritten, faster version of AWG5014.send_waveform_to_list.

    Args:
        w: The waveform
        m1: Marker1
        m2: Marker2
        wfmname: waveform name
    '''
    if wf is None:
        packed_wf = np.zeros(length, dtype='<u2')
    else:
        # Note: we use np.trunc here rather than np.round
        # as it is an order of magnitude faster
        packed_wf = np.trunc(wf * 8191 + 8191.5).astype('<u2')
    if m1 is not None:
        packed_wf += 16384 * m1
    if m2 is not None:
        packed_wf += 32768 * m2

    raw_data = packed_wf.tobytes()

    # if we create a waveform with the same name but different size,
    # it will not get over written
    # Delete the possibly existing file (will do nothing if the file
    # doesn't exist
    s = f'WLISt:WAVeform:DEL "{name}"'
    awg.write(s)

    # create the waveform
    s = f'WLISt:WAVeform:NEW "{name}",{length},INTEGER'
    awg.write(s)

    # upload data
    s1_str = f'WLISt:WAVeform:DATA "{name}",'
    s1 = s1_str.encode('UTF-8')
    s3 = raw_data
    s2_str = '#' + str(len(str(len(s3)))) + str(len(s3))
    s2 = s2_str.encode('UTF-8')

    mes = s1 + s2 + s3
    awg.visa_handle.write_raw(mes)
