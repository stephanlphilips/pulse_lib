"""
Generic data class where all others should be derived from.
"""

from abc import ABC, abstractmethod
import numpy as np
from pulse_lib.segments.utility.segments_c_func import get_effective_point_number

class parent_data(ABC):
    """
        Abstract class hosting some functions that take care of rendering and caching of data and
        makes a template for default functions that are expected in a data object
    """
    start_time = 0
    waveform_cache = dict() 

    @abstractmethod
    def append():
        raise NotImplemented
    
    @abstractmethod
    def slice_time():
        raise NotImplemented
    
    @abstractmethod
    def reset_time(time = None, extend_only = False):
        raise NotImplemented
    
    @abstractmethod
    def get_vmax(self,sample_rate):
        '''
        Calculate the maximum voltage in the current segment_single.
        Args:
            sample_rate (double) :  rate at which is samples (in Hz)
        '''
        raise NotImplemented
    
    @abstractmethod
    def get_vmin(self,sample_rate):
        '''
        Calculate the maximum voltage in the current segment_single.
        Args:
            sample_rate (double) :  rate at which is samples (in Hz)
        '''
        raise NotImplemented
    
    @abstractmethod
    def integrate_waveform(self, pre_delay, post_delay, sample_rate):
        '''
        takes a full integral of the currently scheduled waveform.
        Args:
            start_time (double) : from which points the rendering needs to start
            stop_time (double) : to which point the rendering needs to go (default (-1), to entire segment)
            sample_rate (double) : rate at which the AWG will be run
        Returns:
            integrate (double) : the integrated value of the waveform (unit is mV/sec).
        '''
        raise NotImplemented
    
    @abstractmethod
    def __add__():
        raise NotImplemented
    
    @abstractmethod
    def __mul__():
        raise NotImplemented
    
    @abstractmethod
    def __copy__():
        raise NotImplemented

    @abstractmethod
    def _render(self, sample_rate, pre_delay = 0, post_delay = 0):
        '''
        make a full rendering of the waveform at a predetermined sample rate. This should be defined in the child of this class.
        '''
        raise NotImplemented

    def render(self, pre_delay = 0, post_delay = 0, sample_rate=1e9, clear_cache_on_exit = False):
        '''
        renders pulse
        Args:
            pre_delay (double) : amount of time to put before the sequence the rendering needs to start
            post_delay (double) : to which point in time the rendering needs to go
            sample_rate (double) : rate at which the AWG will be run
            clear_cache_on_exit (bool) : clear the cache on exit (e.g. when you uploaded this waveform to the AWG, remove reference so the garbarge collector can remove it). The ensured low memory occupation.
        returns
            pulse (np.ndarray) : numpy array of the pulse
        '''

        # If no render performed, generate full waveform, we will cut out the right size if needed

        if len(self.waveform_cache) == 0 or self.waveform_cache['sample_rate'] != sample_rate:
            pre_delay_wvf = pre_delay
            if pre_delay > 0:
                pre_delay_wvf = 0
            post_delay_wvf = post_delay
            if post_delay < 0:
                pre_delay_wvf = 0

            self.waveform_cache = {
                'sample_rate' : sample_rate,
                'waveform' : self._render(sample_rate, pre_delay_wvf, post_delay_wvf),
                'pre_delay': pre_delay,
                'post_delay' : post_delay
            }

        # get the waveform
        my_waveform = self.get_resized_waveform(pre_delay, post_delay)
        
        # clear cache if needed
        if clear_cache_on_exit == True:
            self.waveform_cache = dict()

        return my_waveform

    def get_resized_waveform(self, pre_delay, post_delay):
        '''
        extend/shrink an existing waveform
        Args:
            pre_delay (double) : ns to add before
            post_delay (double) : ns to add after the waveform
        Returns:
            waveform (np.ndarray[ndim=1, dtype=double])
        '''

        sample_rate = self.waveform_cache['sample_rate']*1e-9
        sample_time_step = 1/sample_rate

        pre_delay_pt = get_effective_point_number(pre_delay, sample_time_step)
        post_delay_pt = get_effective_point_number(post_delay, sample_time_step)

        wvf_pre_delay_pt = get_effective_point_number(self.waveform_cache['pre_delay'], sample_time_step)
        wvf_post_delay_pt = get_effective_point_number(self.waveform_cache['post_delay'], sample_time_step)

        # points to add/remove from existing waveform
        n_pt_before = - pre_delay_pt + wvf_pre_delay_pt
        n_pt_after = post_delay_pt - wvf_post_delay_pt

        # if cutting is possible (prefered since no copying is involved)
        if n_pt_before <= 0 and n_pt_after <= 0:
            if n_pt_after == 0:
                return self.waveform_cache['waveform'][-n_pt_before:]
            else:
                return self.waveform_cache['waveform'][-n_pt_before: n_pt_after]
        else:
            n_pt = len(self.waveform_cache['waveform']) + n_pt_after + n_pt_before
            new_waveform =  np.zeros((n_pt, ))

            if n_pt_before > 0:
                new_waveform[0:n_pt_before] = self.baseband_pulse_data[0,1]
                if n_pt_after < 0:
                    new_waveform[n_pt_before:] = self.waveform_cache['waveform'][:n_pt_after]
                elif n_pt_after == 0:
                    new_waveform[n_pt_before:] = self.waveform_cache['waveform']
                else:
                    new_waveform[n_pt_before:-n_pt_after] = self.waveform_cache['waveform']
            else:
                new_waveform[:-n_pt_after] = self.waveform_cache['waveform'][-n_pt_before:]

            if n_pt_after > 0:
                new_waveform[-n_pt_after:] =  self.baseband_pulse_data[-1,1]

            return new_waveform

class data_container(np.ndarray):

    def __new__(subtype, input_type=None, shape = (1,)):
        obj = super(data_container, subtype).__new__(subtype, shape, object)
        
        if input_type is not None:
            obj[0] = input_type

        return obj

    @property
    def total_time(self):
        shape = self.shape

        self = self.flatten()
        times = np.empty(self.shape)

        for i in range(len(times)):
            times[i] = self[i].total_time

        self = self.reshape(shape)
        times = times.reshape(shape)
        return times