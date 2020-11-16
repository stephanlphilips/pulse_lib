import time
import numpy as np
import logging
from dataclasses import dataclass, field
from typing import List, Tuple, Dict

from pulse_lib.segments.utility.segments_c_func import get_effective_point_number


class AwgConfig:
    MAX_AMPLITUDE = 1500 # mV
    ALIGNMENT = 10 # waveform must be multiple 10 bytes


class M3202A_Uploader:

    def __init__(self, AWGs, channel_names, channel_map, channel_delays, channel_compensation_limits, channel_attenuation):
        '''
        Initialize the keysight uploader.
        Args:
            AWGs (dict<awg_name,QcodesIntrument>): list with AWG's
            channel_names (list): list with all channel names
            channel_map (dict): dict with channel and AWG+channel location
            channel_delays (dict): channel delays
            channel_compenstation_limits (dict): dict with channel name as key and tuple as value with lower and upper limit
            channel_attenuation (dict): attenuation from AWG to device per channel
        Returns:
            None
        '''
        self.AWGs = AWGs

        self.channel_names = channel_names
        self.channel_map = channel_map
        self.channel_delays = channel_delays
        self.channel_compensation_limits = channel_compensation_limits
        self.channel_attenuation = channel_attenuation

        self.jobs = []
        # hvi is used by scheduler to check whether another hvi must be loaded.
        self.hvi = None


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

        aggregator = UploadAggregator(self.channel_names, self.channel_attenuation, self.channel_compensation_limits, self.channel_delays)

        aggregator.upload_job(job, self.__upload_to_awg)

        self.jobs.append(job)

        duration = time.perf_counter() - start
        logging.info(f'generated upload data ({duration*1000:6.3f} ms)')


    def __upload_to_awg(self, channel_name, waveform):
#        vmin = waveform.min()
#        vmax = waveform.max()
#        length = len(waveform)
#        logging.debug(f'{channel_name}: V({vmin*1000:6.3f}, {vmax*1000:6.3f}) {length}')
        (awg_name, channel) = self.channel_map[channel_name]
        awg = self.AWGs[awg_name]
        wave_ref = awg.upload_waveform(waveform)
        return wave_ref


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

        # queue waveforms
        for channel_name, queue in job.channel_queues.items():
            """
            upload data <tuple>:
                [0] <tuple <double>> : min output voltate, max output voltage
                [1] <list <tuple <mem_loc<int>, n_rep<int>, precaler<int>> : upload locations of differnt segments
                    (by definition backend now merges all segments in 1 since it should
                    not slow you down, but option is left open if this would change .. )
            """
            awg_name, channel_number = self.channel_map[channel_name]

            # This should happen in HVI
            # self.AWGs[awg_name].awg_stop(channel_number)

            self.AWGs[awg_name].set_channel_amplitude(AwgConfig.MAX_AMPLITUDE/1000, channel_number)
            self.AWGs[awg_name].set_channel_offset(0, channel_number)

            # empty AWG queue
            self.AWGs[awg_name].awg_flush(channel_number)

            start_delay = 0 # no start delay
            # Note: Keysight SD1 3.x requires trigger_mode 5 (trigger mode == 1 result in an exception)
            trigger_mode = 5 # software/HVI trigger cycle
            cycles = 1
            for queue_item in queue:
                awg = self.AWGs[awg_name]
                prescaler = awg.convert_sample_rate_to_prescaler(queue_item.sample_rate)
                awg.awg_queue_waveform(
                        channel_number, queue_item.wave_reference,
                        trigger_mode, start_delay, cycles, prescaler)
                trigger_mode = 0 # Auto tigger -- next waveform will play automatically.

        # start hvi
        hw_schedule = job.hw_schedule
        if not hw_schedule.is_loaded():
            hw_schedule.load()
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
        awg_name, channel = next(iter(self.channel_map.values()))
        awg = self.AWGs[awg_name]

        while awg.awg_is_running(channel):
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
    attenuation: float = 1.0
    dc_compensation: bool = False
    dc_compensation_min: float = 0.0
    dc_compensation_max: float = 0.0
    prefix_ns: float = 0.0
    # aggregation state
    integral: float = 0.0
    npt: int = 0
    data: List[np.ndarray] = field(default_factory=list)

@dataclass
class SectionInfo:
    sample_rate: float
    npt: int = 0
    end_time: float = 0.0
    cropped_start: float = 0.0
    cropped_end: float = 0.0
    waveforms: Dict[str,np.ndarray] = field(default_factory=dict, repr=False)
    # timestamps can be generated from sample rates

@dataclass
class JobUploadInfo:
    sections: List[SectionInfo] = field(default_factory=list)
    dc_compensation_duration: float = 0.0
    dc_compensation_voltages: Dict[str, float] = field(default_factory=dict)


class UploadAggregator:
    def __init__(self, channel_names, channel_attenuation, channel_compensation_limits, channel_delays):
        self.npt = 0
        self.channels = dict()

        self.max_prefix_ns = 0
        for channel_name in channel_names:
            info = ChannelInfo()
            self.channels[channel_name] = info

            info.attenuation = channel_attenuation[channel_name]
            info.prefix_ns = -channel_delays[channel_name][0]
            self.max_prefix_ns = max(self.max_prefix_ns, info.prefix_ns)

            if channel_name in channel_compensation_limits:
                # Note: Compensation limits are specified before attenuation, i.e. at AWG output level.
                #       Convert compensation limit to device level.
                info.dc_compensation_min = channel_compensation_limits[channel_name][0] * info.attenuation
                info.dc_compensation_max = channel_compensation_limits[channel_name][1] * info.attenuation
                info.dc_compensation = info.dc_compensation_min < 0 and info.dc_compensation_max > 0


    def upload_job(self, job, awg_upload_func):
        '''
        Steps:
            * add zeros for the amount of channel delay
            * add segments
            *   when sample rate changes add welding samples and upload 'section'
            * append zeros to align all channels
            * add DC compensation
            * add at least 1 zero value and align to muliple of 10 samples
            * upload waveforms

        Welding includes aligment of all channels on a multiple of 10 samples using the highest of the two
        sample rates.
        '''
        self.playback_time = 0
        self.wavelen = 0
        self.upload_delta_ns = 0
        job.upload_info = JobUploadInfo()

        seq0 = job.sequence[0]
        sample_rate = seq0.sample_rate if seq0.sample_rate is not None else job.default_sample_rate
        section_info = SectionInfo(sample_rate)
        job.upload_info.sections.append(section_info)
        self.add_prefix(1e9/sample_rate)

        nseq = len(job.sequence)
        sample_rate_prev = None
        for iseq in range(nseq):
            seq = job.sequence[iseq]

            sample_rate = seq.sample_rate if seq.sample_rate is not None else job.default_sample_rate
            t_sample = 1e9/sample_rate
            if iseq < nseq-1:
                seq_next = job.sequence[iseq+1]
                sample_rate_next = seq_next.sample_rate if seq_next.sample_rate is not None else job.default_sample_rate
            else:
                sample_rate_next = None

            wvf = {}
            integral = {}
            istart = {}
            iend = {}
            len_seq = None
            for channel_name, channel_info in self.channels.items():
                start = time.perf_counter()
                seg = getattr(seq, channel_name)
                wvf[channel_name] = seg.get_segment(job.index, sample_rate)
                integral[channel_name] = seg.integrate(job.index, sample_rate) if job.neutralize else 0
                if len_seq is None:
                    len_seq = len(wvf[channel_name])
                else:
                    if len(wvf[channel_name]) != len_seq:
                        logging.warn(f'waveform {iseq}:{channel_name} {len(wvf)} Sa <> sequence length {len_seq}')
                istart[channel_name] = 0
                iend[channel_name] = len_seq
                duration = time.perf_counter() - start
                logging.debug(f'added {iseq}:{channel_name} {len(wvf[channel_name])} Sa, '
                              f'integral: {integral[channel_name]*1e3:6.2f} uVs in {duration*1000:6.3f} ms')


            # create welding region if sample_rate decreases
            if sample_rate_prev is not None and sample_rate < sample_rate_prev:
                # welding region is length of padding for alignment (delay region is already written)
                nwelding = self.get_wave_alignment()
                ts_welding = 1e9/sample_rate_prev
                crop_start = int(np.round((nwelding*ts_welding - self.upload_delta_ns)/t_sample))

                for channel_name, channel_info in self.channels.items():
                    xseg = wvf[channel_name]
                    nwelding_ch = nwelding + self.get_suffix_length(1e9/sample_rate_prev, channel_info)
                    if np.round(nwelding_ch*ts_welding/t_sample) >= len(xseg):
                        raise Exception(f'segment {iseq} too short for welding. (nwelding_ch:{nwelding_ch} '
                                        f'len_wvf:{len(xseg)})')
                    isub = [np.round((i*ts_welding)/t_sample) for i in np.arange(nwelding_ch)]
                    welding_samples = np.take(xseg, isub)
                    self.add_data(channel_info, welding_samples, 0)

                # welding replaces crop_start samples of segment
                self.add_waveform_time(nwelding, ts_welding, crop_start*t_sample)
                logging.debug(f'welding {iseq-1}->{iseq}: cropping {iseq} with {crop_start*t_sample} ns')

                # next wave
                self.upload_to_awg(job, sample_rate_prev, awg_upload_func)
                self.reset_data(reset_integral=False)
                section_info = SectionInfo(sample_rate)
                section_info.cropped_start = crop_start * t_sample
                job.upload_info.sections.append(section_info)

                # number samples to crop per channel
                for channel_name, channel_info in self.channels.items():
                    istart[channel_name] = crop_start + self.get_suffix_length(t_sample, channel_info)

                len_seq -= crop_start


            # create welding region if sample rate increases
            if sample_rate_next is not None and sample_rate_next > sample_rate:
                ts_welding = 1e9/sample_rate_next
                # cut away overlapping delay region and align. Use maximum needed for welding
                npad = int(np.ceil(self.get_max_prefix(ts_welding)*ts_welding/t_sample))
                len_seq -= npad
                # wavelength alignment:
                waveremainder = self.get_wave_remainder(len_seq)
                len_seq -= waveremainder

                crop_end = npad + waveremainder
                section_info.cropped_end = crop_end * t_sample

                for channel_name, channel_info in self.channels.items():
                    xseg = wvf[channel_name]
                    iend[channel_name] = len(xseg)-crop_end + self.get_suffix_length(t_sample, channel_info)
                    samples = xseg[istart[channel_name]:iend[channel_name]]
                    self.add_data(channel_info, samples, integral[channel_name])

                self.add_waveform_time(len_seq, t_sample, len_seq*t_sample)
                logging.debug(f'welding {iseq}->{iseq+1}: cropping {iseq} with {crop_end*t_sample} ns')

                # next wave
                self.upload_to_awg(job, sample_rate, awg_upload_func)
                self.reset_data(reset_integral=False)
                section_info = SectionInfo(sample_rate_next)
                job.upload_info.sections.append(section_info)

                nwelding = int(np.round((crop_end*t_sample - self.upload_delta_ns)/ts_welding))

                for channel_name, channel_info in self.channels.items():
                    xseg = wvf[channel_name]
                    nwelding_ch = nwelding - self.get_suffix_length(1e9/sample_rate_next, channel_info)
                    if nwelding_ch * ts_welding > len(xseg)*t_sample:
                        raise Exception('segment too short for welding')
                    isub = [min(len(xseg)-1, iend[channel_name] +  np.round((i*ts_welding)/t_sample)) for i in np.arange(nwelding_ch)]
                    welding_samples = np.take(xseg, isub)
                    self.add_data(channel_info, welding_samples, 0)

                self.add_waveform_time(nwelding, ts_welding, crop_end*t_sample)

            else:
                for channel_name, channel_info in self.channels.items():
                    xseg = wvf[channel_name]
                    samples = xseg[istart[channel_name]:]
                    self.add_data(channel_info, samples, integral[channel_name])

                self.add_waveform_time(len_seq, t_sample, len_seq*t_sample)

            sample_rate_prev = sample_rate

        self.add_suffix(t_sample)

        if job.neutralize:
            self.add_dc_compensation(sample_rate, job.upload_info)

        self.append_zeros(sample_rate)

        self.upload_to_awg(job, sample_rate, awg_upload_func)

        job.playback_time = self.playback_time

        self.reset_data()


    def upload_to_awg(self, job, sample_rate, awg_upload_func):

        section_info = job.upload_info.sections[-1]
        section_info.npt = self.npt
        section_info.end_time = self.playback_time

        for channel_name, channel_info in self.channels.items():

            waveform = self.get_upload_data(channel_info)

            wave_ref = awg_upload_func(channel_name, waveform)
            job.add_waveform(channel_name, wave_ref, sample_rate)
            section_info.waveforms[channel_name] = waveform


    def get_max_prefix(self, t_sample):
        return int(np.round(self.max_prefix_ns/t_sample))


    def get_prefix_length(self, t_sample, channel_info):
        return int(np.round(channel_info.prefix_ns/t_sample))


    def get_suffix_length(self, t_sample, channel_info):
        return self.get_max_prefix(t_sample) - self.get_prefix_length(t_sample, channel_info)


    def add_prefix(self, t_sample):
        for channel_name, channel_info in self.channels.items():
            wvf = np.zeros(self.get_prefix_length(t_sample, channel_info))
            self.add_data(channel_info, wvf, 0.0)
        self.add_waveform_time(self.get_max_prefix(t_sample), t_sample, self.max_prefix_ns)


    def add_suffix(self, t_sample):
        wavepadding = self.get_wave_alignment()
        for channel_name, channel_info in self.channels.items():
            wvf = np.zeros(wavepadding + self.get_suffix_length(t_sample, channel_info))
            self.add_data(channel_info, wvf, 0.0)

        self.add_waveform_time(wavepadding, t_sample, wavepadding*t_sample)


    def get_wave_alignment(self, n=0):
        return (AwgConfig.ALIGNMENT - 1) - (self.wavelen + n - 1) % AwgConfig.ALIGNMENT


    def get_wave_remainder(self, n):
        return (self.wavelen + n) % AwgConfig.ALIGNMENT


    def reset_data(self, reset_integral=True):
        self.npt = 0
        for channel_info in self.channels.values():
            channel_info.data = []
            channel_info.npt = 0
            if reset_integral:
                channel_info.integral = 0.0


    def add_data(self, channel_info, wvf, integral):
        if len(wvf) == 0:
            return
        channel_info.integral += integral
        channel_info.data.append(wvf)
        channel_info.npt += len(wvf)
        self.npt = max(self.npt, channel_info.npt)


    def add_waveform_time(self, n, sampling_period, duration):
        self.wavelen += n
        self.playback_time += n * sampling_period
        self.upload_delta_ns += n * sampling_period - duration


    def add_dc_compensation(self, sample_rate, upload_info):

        compensation_time = self.get_max_compensation_time()
        compensation_npt = int(np.ceil(compensation_time * sample_rate))

        logging.debug(f'DC compensation: {compensation_npt} samples')

        for channel_name, channel_info in self.channels.items():

            if compensation_npt > 0 and channel_info.dc_compensation:
                compensation_voltage = -channel_info.integral * sample_rate / compensation_npt
                dc_compensation = np.full((compensation_npt,), compensation_voltage)
                self.add_data(channel_info, dc_compensation, -channel_info.integral)
                upload_info.dc_compensation_voltages[channel_name] = compensation_voltage
                logging.debug(f'DC compensation {channel_name}: {compensation_voltage:6.1f} mV {compensation_npt} Sa')
            else :
                no_compensation = np.zeros((compensation_npt,), np.float)
                self.add_data(channel_info, no_compensation, 0)
                upload_info.dc_compensation_voltages[channel_name] = 0

        self.add_waveform_time(compensation_npt, 1e9/sample_rate, compensation_npt*1e9/sample_rate)
        upload_info.dc_compensation_duration = compensation_npt*1e9/sample_rate


    def append_zeros(self, sample_rate):
        ''' Align and add extra zeros to make sure you end up with 0V when done.'''
        min_zeros = 1
        wavepadding = self.get_wave_alignment(min_zeros)
        zero_padding = np.zeros(wavepadding+min_zeros, dtype=np.float)
        for channel_info in self.channels.values():
            self.add_data(channel_info, zero_padding, 0)

        self.add_waveform_time(AwgConfig.ALIGNMENT, 1e9/sample_rate, AwgConfig.ALIGNMENT*1e9/sample_rate)


    def get_upload_data(self, channel_info):

        data = channel_info.data
        # concat all
        # divide by attenuation
        # divide by AwgConfig.AWG_AMPLITUDE
        waveform = np.concatenate(data)
        # note: numpy inplace multiplication is much faster than standard multiplication
        waveform *= 1/(channel_info.attenuation * AwgConfig.MAX_AMPLITUDE)
        return waveform


    def get_max_compensation_time(self):
        '''
        generate a DC compensation of the pulse.
        As usuallly we put capacitors in between the AWG and the gate on the sample, you need to correct
        for the fact that the low fequencies are not present in your transfer function.
        This can be done simply by making the total integral of your function 0.

        Args:
            sample_rate (float) : rate at which the AWG runs.
        '''
        max_compensation_time = 0
        for channel_info in self.channels.values():
            compensation_time = self.get_compensation_time(channel_info)
            max_compensation_time = max(max_compensation_time, compensation_time)

        return max_compensation_time


    def get_compensation_time(self, channel_info):
        '''
        return the minimal compensation time that is needed.
        Returns:
            compensation_time : minimal duration that is needed for the voltage compensation
        '''
        if not channel_info.dc_compensation:
            return 0

        if channel_info.integral <= 0:
            return -channel_info.integral / channel_info.dc_compensation_max
        else:
            return -channel_info.integral / channel_info.dc_compensation_min


