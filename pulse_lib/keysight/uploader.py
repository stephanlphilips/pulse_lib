
import time
import logging
import numpy as np
from pulse_lib.keysight.uploader_core.uploader import waveform_cache_container
from pulse_lib.segments.utility.segments_c_func import get_effective_point_number


class keysight_uploader():
    """
    Object responsible for uploading waveforms to the keysight AWG in a timely fashion.
    """
    def __init__(self, AWGs, cpp_uploader,channel_names, channel_locations, channel_delays, channel_compenstation_limits, AWG_to_dac_ratio):
        '''
        Initialize the keysight uploader.
        Args:
            AWGs (dict<awg_name,QcodesIntrument>) : list with AWG's
            cpp_uploader (keysight_upload_module) : class that performs normalisation and conversion of the wavorm to short + upload.
            channel_names(list) : list with all the names of the channels
            channel_locations (dict): dict with channel and AWG+channel location
            channel_compenstation_limits (dict) : dict with channel name as key and tuple as value with lower and upper limit
        Returns:
            None
        '''
        self.memory_allocation = dict()
        # TODO reinit memory on start-up
        self.AWGs = AWGs
        self.current_HVI = None
        self.current_HVI_ID = None
        self.cpp_uploader = cpp_uploader
        self.channel_names = channel_names
        self.channel_map = channel_locations
        self.channel_delays = channel_delays
        self.channel_compenstation_limits = channel_compenstation_limits
        self.AWG_to_dac_ratio = AWG_to_dac_ratio
        self.upload_queue = []
        self.upload_ready_to_start = []
        self.upload_done = []

    def create_job(self, sequence, index, seq_id, n_rep, sample_rate, neutralize=True):
        return upload_job(sequence, index, seq_id, n_rep, sample_rate, neutralize)

    def add_upload_job(self, job):
        '''
        add a job to the uploader.
        Args:
            job (upload_job) : upload_job object that defines what needs to be uploaded and possible post processing of the waveforms (if needed)
        '''
        self.upload(job)

    def __get_upload_data(self, seq_id, index):
        """
        get job data of an uploaded segment
        Args:
            seq_id (uuid) : id of the sequence
            index (tuple) : index that has to be played
        Return:
            job (upload_job) :job, with locations of the sequences to be uploaded.
        """
        # check if job if job is uploaded.
        for j in range(5):
            for i in range(len(self.upload_ready_to_start)):
                job = self.upload_ready_to_start[i]
                if job.id == seq_id and job.index == index:
                    return self.upload_ready_to_start.pop(i)

        raise ValueError("Sequence with id {}, index {} not placed for upload .. . Always make sure to first upload your segment and then do the playback.")

    def _segment_AWG_memory(self):
        '''
        Generates segments in the memory in the Keysight AWG.
        '''
        self.cpp_uploader.resegment_memory()

        # set to single shot meaurements. This is the default option for HVI based code.
        for channel, channel_loc in self.channel_map.items():
            self.awg[channel_loc[0]].awg_queue_config(channel_loc[1], 0)

    def play(self, seq_id, index, release = True):
        """
        start playback of a sequence that has been uploaded.
        Args:
            seq_id (uuid) : id of the sequence
            index (tuple) : index that has to be played
            release (bool) : release memory on AWG after done.
        """

        """
        steps :
        0) get upload data (min max voltages for all the channels, total time of the sequence, location where things are stored in the AWG memory.) and wait until the AWG is idle
        1) set voltages for all the channels.
        2) make queue for each channels (now assuming single waveform upload).
        3) upload HVI code & start.
        """
        # 0)
        job =  self.__get_upload_data(seq_id, index)
        self.wait_until_AWG_idle()

        # 1 + 2)
        # flush the queue's
        for channel_name, data in job.upload_data.items():
            """
            upload data <tuple>:
                [0] <tuple <double>> : min output voltate, max output voltage
                [1] <list <tuple <mem_loc<int>, n_rep<int>, precaler<int>> : upload locations of differnt segments
                    (by definition backend now merges all segments in 1 since it should
                    not slow you down, but option is left open if this would change .. )
            """
            awg_name, channel_number = self.channel_map[channel_name.decode('ascii')]
            v_pp, v_off = convert_min_max_to_vpp_voff(*data[0])

            # This should happen in HVI
            # self.AWGs[awg_name].awg_stop(channel_number)

            self.AWGs[awg_name].set_channel_amplitude(v_pp/1000/2,channel_number) #amp = vpp/2 (speciefied in V on module, so therefore factor 1000)
            self.AWGs[awg_name].set_channel_offset(v_off/1000,channel_number)

            self.AWGs[awg_name].awg_flush(channel_number)

            start_delay = 0 # no start delay
            trigger_mode = 1 # software/HVI trigger
            cycles = 1
            prescaler = job.prescaler
            for segment_number in data[1]:
                self.AWGs[awg_name].awg_queue_waveform(channel_number,segment_number,trigger_mode,start_delay,cycles,prescaler)
                trigger_mode = 0 # Auto tigger -- next waveform will play automatically.
        # 3)
        if job.HVI_start_function is None:
            job.HVI.load()
            job.HVI.start()
        else:
            job.HVI_start_function(job.HVI, self.AWGs, self.channel_map, job.playback_time, job.n_rep, **job.HVI_kwargs)

        if release == True:
            self._release_memory_jobs()
            self.upload_done.append(job)
        else:
            # return job to queue for reuse of waveforms
            self.upload_ready_to_start.append(job)


    def release_memory(self, seq_id=None, index=None):
        for job in self.upload_ready_to_start:
            if (seq_id is None
                or (job.seq_id == seq_id and (index is None or job.index == index))):
                self.upload_done.append(job)

        self._release_memory_jobs()


    def _release_memory_jobs(self):
        # release the memory of all jobs that are uploaded. Be careful to do not run this when active playback is happening. Otherwise you risk of overwriting a waveform while playing.
        for job in self.upload_done:
            self.cpp_uploader.release_memory(job.waveform_cache)
        self.upload_done = []


    def upload(self, job):
        '''
        Class taking care of putting the waveform on the right AWG. This is a continuous thread that is run in the background.

        Steps:
        1) get all the upload data
        2) perform DC correction (if needed)
        3) compile the HVI script for the next upload
        5a) convert data in an aprropriate upload format (c++)
        5b) upload all data (c++)
        6) write in the job object the resulting locations of sequences that have been uploaded.

        '''

        start = time.perf_counter()
        logging.debug('uploading')

        # 1) get all the upload data -- construct object to hall the rendered data
        waveform_cache = waveform_cache_container(self.channel_map, self.channel_compenstation_limits)

        sample_rate = job.sample_rate

        for i in range(len(job.sequence)):

            seg = job.sequence[i]

            for channel in self.channel_names:

                wvf = seg.get_waveform(channel, job.index, sample_rate)
                integral = 0
                if job.neutralize == True:
                    integral = getattr(seg, channel).integrate(job.index, sample_rate)

                vmin = getattr(seg, channel).v_min(job.index, sample_rate)
                vmax = getattr(seg, channel).v_max(job.index, sample_rate)

                if channel in self.AWG_to_dac_ratio.keys(): #start Luca modification
                    ratio = self.AWG_to_dac_ratio[channel]
                else:
                    ratio = 1 #end Luca modification

                if i == 0:
                    pre_delay = self.channel_delays[channel][0]
                    v = wvf[0]
                    pre_delay_pt = get_effective_point_number(-pre_delay, 1e9/sample_rate)
                    pre_delay_wvf = v*np.ones(pre_delay_pt)
                    waveform_cache[channel].add_data(pre_delay_wvf/ratio, (v/ratio, v/ratio), v*pre_delay_pt*1e-9/ratio)

                waveform_cache[channel].add_data(wvf/ratio, (vmin/ratio, vmax/ratio), integral/ratio)

                if i == len(job.sequence) -1:
                    post_delay = self.channel_delays[channel][1]
                    v = wvf[-1]
                    post_delay_pt = get_effective_point_number(post_delay, 1e9/sample_rate)
                    post_delay_wvf = v*np.ones(post_delay_pt)
                    waveform_cache[channel].add_data(post_delay_wvf/ratio, (v/ratio, v/ratio), v*post_delay_pt*1e-9/ratio)


        # 2) perform DC correction (if needed)
        '''
        Steps: [TODO : best way to include sample rate here? (by default now 1GS/s)]
            a) calculate total compensation time needed (based on given boundaries).
            b) make sure time is modulo 10 (do that here?)
            c) add segments with the compenstated pulse for the given total time.
        '''
        waveform_cache.generate_DC_compenstation(sample_rate)
        # TODO express this in time instead of points (now assumed one ns is point in the AWG (not very robust..))
        job.waveform_cache = waveform_cache
        if job.prescaler == 0:
            job.playback_time = waveform_cache.npt
        elif job.prescaler == 1:
            job.playback_time = waveform_cache.npt*5*job.prescaler
        else:
            job.playback_time = waveform_cache.npt*5*job.prescaler*2

        # 3)
        if job.HVI is not None:
            job.compile_HVI()

        duration = time.perf_counter() - start
        logging.info(f'generated wavefroms in {duration*1000:6.3f} ms')

        start = time.perf_counter()
        # 3 + 4a+b)
        job.upload_data = self.cpp_uploader.add_upload_data(waveform_cache)

        # submit the current job as completed.
        self.upload_ready_to_start.append(job)

        duration = time.perf_counter() - start
        logging.info(f'uploaded in {duration*1000:6.3f} ms')


    def wait_until_AWG_idle(self):
        '''
        check if the AWG is doing playback, when done, release this function
        '''
        # assume all awg's are used and also all the channels
        awg_name, channel = next(iter(self.channel_map.values()))
        awg = self.AWGs[awg_name]

        while awg.awg.AWGisRunning(channel):
            time.sleep(0.001)


class upload_job(object):
    """docstring for upload_job"""
    def __init__(self, sequence, index, seq_id, n_rep, sample_rate, neutralize=True, priority=0):
        '''
        Args:
            sequence (list of segment_container): list with segment_containers in sequence
            index (tuple) : index that needs to be uploaded
            seq_id (uuid) : id of the sequence
            n_rep (int) : number of repetitions of this sequence.
            neutralize (bool) : place a neutralizing segment at the end of the upload
            priority (int) : priority of the job (the higher one will be excuted first)
        '''
        self.sequence = sequence
        self.id = seq_id
        self.index = index
        self.n_rep = n_rep
        self.sample_rate = sample_rate
        self.prescaler = convert_sample_rate_to_prescaler(sample_rate)
        self.neutralize = neutralize
        self.priority = priority
        self.playback_time = 0 #total playtime of the waveform
        self.upload_data = None
        self.waveform_cache = None
        self.HVI = None

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
        self.HVI_compile_function = compile_function
        self.HVI_start_function = start_function
        self.HVI_kwargs = kwargs

    def compile_HVI(self):
        self.HVI_compile_function(self.HVI, self.playback_time, self.n_rep, **self.HVI_kwargs)


def convert_min_max_to_vpp_voff(v_min, v_max):
    # vpp = v_max - v_min
    # voff = (v_min + v_max)/2
    voff = 0
    vpp = max(abs(v_min), abs(v_max))*2
    return vpp, voff

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

def convert_prescaler_to_sample_rate(prescalor):
    """
    Keysight specific function.

    Args:
        prescalor (int) : prescalor set to the awg.

    Returns:
        sample_rate (float) : effective sample rate the AWG will be running
    """
    if prescalor == 0:
        return 1e9
    if prescalor == 1:
        return 200e6
    else:
        return 1e9/(2*5*prescalor)