
import logging

from .hardware_schedule import HardwareSchedule

# function prototypes of old style HVI:
# def load_HVI(AWGs, channel_map, *args,**kwargs):
# def excute_HVI(HVI, AWGs, channel_map, playback_time, n_rep, *args, **kwargs):


class HviCompatibilityWrapper(HardwareSchedule):
    verbose = False
    loaded_schedule = None

    def __init__(self, HVI_ID, AWGs, channel_map, load_HVI, execute_HVI):
        self.awgs = AWGs
        self.channel_map = channel_map
        self.load_hvi = load_HVI
        self.execute_hvi = execute_HVI
        self.hvi = None
        self.schedule_parms = {}
        self.hvi_id = HVI_ID

    def set_schedule_parameters(self, **kwargs):
        for key,value in kwargs.items():
            self.schedule_parms[key] = value

    def compile(self):
        # compilation is included in load_hvi
        pass

    def load(self):
        if self.hvi is None:
            self.hvi = self.load_hvi(self.awgs, self.channel_map, **self.schedule_parms)
        HviCompatibilityWrapper.loaded_schedule = self.hvi

    def is_loaded(self):
        return self.hvi is not None

    def unload(self):
        self.close()

    def start(self, waveform_duration, n_repetitions, sequence_parameters):
        hvi_params = {**self.schedule_parms, **sequence_parameters}
        if self.verbose:
            logging.debug(f'start: {hvi_params}')
        self.execute_hvi(self.hvi, self.awgs, self.channel_map, waveform_duration, n_repetitions, **hvi_params)

    def is_running(self):
        for (awg_name, channel) in self.channel_map.items():
            awg = self.awgs[awg_name]
            return awg.awg_is_running(channel)

    def close(self):
        self.hvi.releaseHW()
        self.hvi.close()
        self.hvi = None
        self.hvi_id = '<not loaded>'
        HviCompatibilityWrapper.loaded_schedule = None

    def __del__(self):
        if self.hvi is not None and HviCompatibilityWrapper.loaded_schedule == self.hvi:
            try:
                logging.warn(f'Automatic close of HVI in __del__()')
                self.close()
            except:
                logging.error(f'Exception closing HVI', exc_info=True)

## sequencer:
#    def add_HVI(self, HVI_ID ,HVI_to_load, compile_function, start_function, **kwargs):
#        '''
#        Add HVI code to the AWG.
#        Args:
#            HVI_ID (str) : string that gives an ID to the HVI that is currently loaded.
#            HVI_to_load (function) : function that returns a HVI file.
#            compile_function (function) : function that compiles the HVI code. Default arguments that will be provided are (HVI, npt, n_rep) = (HVI object, number of points of the sequence, number of repetitions wanted)
#            start_function (function) : function to be executed to start the HVI (this can also be None)
#            kwargs : keyword arguments for the HVI script (see usage in the examples (e.g. when you want to provide your digitzer card))
#        '''
#        if self.uploader.current_HVI_ID != HVI_ID:
#            self.HVI = HVI_to_load(self.uploader.AWGs, self.uploader.channel_map, **kwargs)
#            self.uploader.current_HVI_ID = HVI_ID
#            self.uploader.current_HVI = self.HVI
#        else:
#            self.HVI = self.uploader.current_HVI
#
#        self.HVI_compile_function = compile_function
#        self.HVI_start_function = start_function
#        self.HVI_kwargs = kwargs
#
#    def upload():
#        if self.HVI is not None:
#            upload_job.add_HVI(self.HVI, self.HVI_compile_function, self.HVI_start_function, **{**self.HVI_kwargs, **self._HVI_variables.item(tuple(index)).HVI_markers})
#
#####
#            job.HVI_start_function(job.HVI, self.AWGs, self.channel_map, job.playback_time, job.n_rep, **job.HVI_kwargs)
