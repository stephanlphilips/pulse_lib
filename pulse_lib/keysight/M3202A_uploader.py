import time
import numpy as np
import logging
from dataclasses import dataclass, field
from typing import List, Tuple

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
        self.current_HVI = None
        self.current_HVI_ID = None

        self.channel_names = channel_names
        self.channel_map = channel_map
        self.channel_delays = channel_delays
        self.channel_compensation_limits = channel_compensation_limits
        self.channel_attenuation = channel_attenuation

        self.jobs = []


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

        """
        steps :
        0) get upload data (min max voltages for all the channels, total time of the sequence, location where things are stored in the AWG memory.) and wait until the AWG is idle
        1) set voltages for all the channels.
        2) make queue for each channels (now assuming single waveform upload).
        3) upload HVI code & start.
        """
        # 0)
        job =  self.__get_job(seq_id, index)
        self.wait_until_AWG_idle()

        # 1 + 2)
        # flush the queue's
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

            self.AWGs[awg_name].set_channel_amplitude(AwgConfig.MAX_AMPLITUDE/1000,channel_number)
            self.AWGs[awg_name].set_channel_offset(0,channel_number)

            # empty AWG queue
            self.AWGs[awg_name].awg_flush(channel_number)

            start_delay = 0 # no start delay
            trigger_mode = 1 # software/HVI trigger
            cycles = 1
            for queue_item in queue:
                self.AWGs[awg_name].awg_queue_waveform(
                        channel_number, queue_item.wave_reference,
                        trigger_mode,start_delay,cycles, queue_item.prescaler)
                trigger_mode = 0 # Auto tigger -- next waveform will play automatically.

        # 3)
        if job.HVI_start_function is None:
            job.HVI.load()
            job.HVI.start()
        else:
            job.HVI_start_function(job.HVI, self.AWGs, self.channel_map, job.playback_time, job.n_rep, **job.HVI_kwargs)


        # determine if the current job needs to be reused.
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
    prescaler: int


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
        self.HVI = None
        logging.debug(f'new job {seq_id}-{index}')


    def add_HVI(self, HVI, compile_function, start_function, **kwargs):
        """
        Introduce HVI functionality to the upload.
        args:
            HVI (SD_HVI) : HVI object from the keysight libraries
            compile_function (function) : function that compiles the HVI code. Default arguments that will be provided are (HVI, npt, n_rep) = (HVI object, number of points of the sequence, number of repetitions wanted)
            start_function (function) :function to be executed to start the HVI (this can also be None)
            kwargs : keyword arguments for the HVI script (see usage in the examples (e.g. when you want to provide your digitzer card))
        """
        self.HVI = HVI
        self.HVI_start_function = start_function
        self.HVI_kwargs = kwargs


    def add_waveform(self, channel_name, wave_ref, prescaler):
        if channel_name not in self.channel_queues:
            self.channel_queues[channel_name] = []

        self.channel_queues[channel_name].append(AwgQueueItem(wave_ref, prescaler))


    def release(self):
        if self.released:
            logging.warning(f'job {self.seq_id}-{self.index} already released')
            return

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
    attenuation: float = 1.0
    dc_compensation: bool = False
    dc_compensation_min: float = 0.0
    dc_compensation_max: float = 0.0
    channel_delays: Tuple[float] = (0,0)
    integral: float = 0.0
    npt: int = 0
    data: List[List[float]] = field(default_factory=list)


class UploadAggregator:
    def __init__(self, channel_names, channel_attenuation, channel_compensation_limits, channel_delays):
        self.npt = 0
        self.channels = dict()

        for channel_name in channel_names:
            info = ChannelInfo()
            self.channels[channel_name] = info

            info.attenuation = channel_attenuation[channel_name]
            info.channel_delays = channel_delays[channel_name]

            if channel_name in channel_compensation_limits:
                # Note: Compensation limits are specified before attenuation, i.e. at AWG output level.
                #       Convert compensation limit to device level.
                info.dc_compensation_min = channel_compensation_limits[channel_name][0] * info.attenuation
                info.dc_compensation_max = channel_compensation_limits[channel_name][1] * info.attenuation
                info.dc_compensation = info.dc_compensation_min < 0 and info.dc_compensation_max > 0


    def upload_job(self, job, awg_upload_func):
        '''
        Steps:
        1) get all the upload data
        2) perform DC correction (if needed)
        3) convert data in an aprropriate upload format
        4) start upload of all data
        5) store reference to uploaded waveform in job
        '''
        current_sample_rate = None
        job.playback_time = 0

        for i in range(len(job.sequence)):

            add_pre_delay = i == 0
            add_post_delay = i == len(job.sequence) -1
            # add post delay when there is a sample rate change
            if i < len(job.sequence) - 1 and job.sequence[i+1].sample_rate != job.sequence[i].sample_rate:
                add_post_delay = True

            seg = job.sequence[i]

            sample_rate = seg.sample_rate if seg.sample_rate is not None else job.default_sample_rate
            if current_sample_rate is not None and current_sample_rate != sample_rate:
                self.align_data(sample_rate, keep_voltage=True)
                self.upload_to_awg(job, current_sample_rate, awg_upload_func)
                job.playback_time += self.npt / current_sample_rate * 1e9
                self.reset_data(reset_integral=False)
                logging.info(f'playback time:{job.playback_time}')
                add_pre_delay = True

            current_sample_rate = sample_rate


            for channel_name, channel_info in self.channels.items():
                start = time.perf_counter()

                pre_delay = 0
                post_delay = 0

                wvf = seg.get_waveform(channel_name, job.index, sample_rate)
                integral = 0
                if job.neutralize:
                    integral = getattr(seg, channel_name).integrate(job.index, sample_rate)

                if add_pre_delay and channel_info.channel_delays[0] < 0:
                    pre_delay = channel_info.channel_delays[0]
                    v = wvf[0] if len(wvf) > 0 else 0
                    pre_delay_pt = -get_effective_point_number(pre_delay, 1e9/sample_rate)
                    pre_delay_wvf = v*np.ones(pre_delay_pt)
                    self.add_data(channel_info, pre_delay_wvf, v*pre_delay_pt*1e-9)

                self.add_data(channel_info, wvf, integral)

                if add_post_delay and channel_info.channel_delays[1] > 0:
                    post_delay = channel_info.channel_delays[1]
                    v = wvf[-1] if len(wvf) > 0 else 0
                    post_delay_pt = get_effective_point_number(post_delay, 1e9/sample_rate)
                    post_delay_wvf = v*np.ones(post_delay_pt)
                    self.add_data(channel_info, post_delay_wvf, v*post_delay_pt*1e-9)

                duration = time.perf_counter() - start
                logging.debug(f'added {i}:{channel_name} {duration*1000:6.3f} ms {len(wvf)} Sa, integral: {integral}, '
                              f'pre:{pre_delay}, post:{post_delay}')

        if job.neutralize:
            self.add_dc_compensation(sample_rate)

        self.align_data(sample_rate)
        self.append_zeros()

        self.upload_to_awg(job, current_sample_rate, awg_upload_func)

        job.playback_time += self.npt / current_sample_rate * 1e9
        logging.info(f'playback time:{job.playback_time}')

        self.reset_data()


    def upload_to_awg(self, job, sample_rate, awg_upload_func):

        for channel_name, channel_info in self.channels.items():

            waveform = self.get_upload_data(channel_info)

            wave_ref = awg_upload_func(channel_name, waveform)
            prescaler = convert_sample_rate_to_prescaler(sample_rate)
            job.add_waveform(channel_name, wave_ref, prescaler)


    def reset_data(self, reset_integral=True):
        self.npt = 0
        for channel_info in self.channels.values():
            channel_info.data = []
            channel_info.npt = 0
            if reset_integral:
                channel_info.integral = 0.0


    def add_data(self, channel_info, wvf, integral):
        channel_info.integral += integral
        channel_info.data.append(wvf)
        channel_info.npt += len(wvf)
        self.npt = max(self.npt, channel_info.npt)


    def add_dc_compensation(self, sample_rate):

        compensation_npt = self.get_compensation_npt(sample_rate)

        for channel_name, channel_info in self.channels.items():

            if compensation_npt > 0 and channel_info.dc_compensation:
                compensation_voltage = -channel_info.integral * sample_rate / compensation_npt
                dc_compensation = np.full((compensation_npt,), compensation_voltage)
                self.add_data(channel_info, dc_compensation, -channel_info.integral)
                logging.debug(f'DC compensation {channel_name}: {compensation_voltage:6.1f} mV {compensation_npt} Sa')
            else :
                no_compensation = np.zeros((compensation_npt,), np.float)
                self.add_data(channel_info, no_compensation, 0)


    def align_data(self, sample_rate, keep_voltage=False):

        npt = self.npt
        remainder = npt % AwgConfig.ALIGNMENT
        padding_npt = (AwgConfig.ALIGNMENT - remainder) if remainder > 0 else 0
        total_npt = npt + padding_npt

        for channel_name, channel_info in self.channels.items():
            if channel_info.npt != npt:
                logging.warn(f'Unequal data length {channel_name}:{channel_info.npt} overall:{self.npt}')

            npt_fill = total_npt - channel_info.npt

            voltage = 0
            if keep_voltage:
                voltage = channel_info.data[-1][-1]

            padding_samples = np.full((npt_fill,), voltage)
            self.add_data(channel_info, padding_samples, voltage * padding_npt / sample_rate)


    def append_zeros(self):
        ''' Adds extra zeros to make sure you end up with 0V when done.'''
        zero_padding = np.zeros(AwgConfig.ALIGNMENT, dtype=np.float)
        for channel_info in self.channels.values():
            self.add_data(channel_info, zero_padding, 0)


    def get_upload_data(self, channel_info):

        data = channel_info.data
        # concat all
        # divide by attenuation
        # divide by AwgConfig.AWG_AMPLITUDE
        waveform = np.concatenate(data)
        # note: numpy inplace multiplication is much faster than standard multiplication
        waveform *= 1/(channel_info.attenuation * AwgConfig.MAX_AMPLITUDE)
        return waveform


    def get_compensation_npt(self, sample_rate):
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

        compensation_npt = int(max_compensation_time * sample_rate + 0.99)

        logging.debug(f'DC compensation: {compensation_npt} samples')

        return compensation_npt


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


def convert_sample_rate_to_prescaler(sample_rate):
    """
    Keysight specific function.

    Args:
        sample_rate (float) : sample rate
    Returns:
        prescaler (int) : prescaler set to the awg.
    """
    if sample_rate > 200e6:
        prescaler = 0
    elif sample_rate > 50e6:
        prescaler = 1
    else:
        prescaler = int(1e9/(5*sample_rate*2))

    return prescaler


def get_effective_sample_rate(sample_rate):
    """
    Returns the sample rate that will be used by the Keysight AWG.
    This is the a rate >= requested sample rate.
    """
    return convert_prescaler_to_sample_rate(convert_sample_rate_to_prescaler(sample_rate))


def convert_prescaler_to_sample_rate(prescaler):
    """
    Keysight specific function.

    Args:
        prescaler (int) : prescaler set to the awg.

    Returns:
        sample_rate (float) : effective sample rate the AWG will be running
    """
    if prescaler == 0:
        return 1e9
    if prescaler == 1:
        return 200e6
    else:
        return 1e9/(2*5*prescaler)
