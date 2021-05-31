"""
Generic data class where all others should be derived from.
"""
import uuid
from abc import ABC, abstractmethod
import numpy as np
from pulse_lib.segments.data_classes.lru_cache import LruCache

import copy

class parent_data(ABC):
    """
        Abstract class hosting some functions that take care of rendering and caching of data and
        makes a template for default functions that are expected in a data object
    """
    start_time = 0
    software_marker_data = dict()

    waveform_cache = LruCache(100)

    def __init__(self):
        self.id = uuid.uuid4()

    @classmethod
    def set_waveform_cache_size(cls, size):
        '''
        Set the new (maximum) size of the waveform cache.
        The cache is cleared when its size changes.
        '''
        if size != cls.waveform_cache.max_size:
            cls.waveform_cache = LruCache(size)

    @classmethod
    def clear_waveform_cache(cls):
        '''
        Clears the waveform cache (freeing memory).
        '''
        # clear the cache by initializing a new one of the same size
        cls.waveform_cache = LruCache(cls.waveform_cache.max_size)

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
    def wait(time):
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
    def integrate_waveform(self, sample_rate):
        '''
        takes a full integral of the currently scheduled waveform.
        Args:
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
    def _render(self, sample_rate, ref_channel_states):
        '''
        make a full rendering of the waveform at a predetermined sample rate. This should be defined in the child of this class.
        '''
        raise NotImplemented

    def add_software_marker(self, marker_name, time):
        '''
        add a marker in software (used as arguments for HVI commands)

        Args:
            marker_name (str) : name of the maker
            time (double) : time in ns where to apply the marker
        '''
        self.software_marker_data[marker_name] = time

    def render(self, sample_rate=1e9, ref_channel_states=None):
        '''
        renders pulse
        Args:
            sample_rate (double) : rate at which the AWG will be run
        returns
            pulse (np.ndarray) : numpy array of the pulse
        '''
        # Render only when there is no matching cached waveform
        cache_entry = self._get_cached_data_entry()

        if (cache_entry.data is None
            or cache_entry.data['sample_rate'] != sample_rate
            or cache_entry.data['ref_states'] != ref_channel_states):
            waveform = self._render(sample_rate, ref_channel_states)
            cache_entry.data = {
                'sample_rate' : sample_rate,
                'waveform' : waveform,
                'ref_states' : ref_channel_states
            }
        else:
            waveform = cache_entry.data['waveform']

        return waveform

    def _get_cached_data_entry(self):
        return self.waveform_cache[self.id]


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

    @property
    def start_time(self):
        shape = self.shape

        self = self.flatten()
        times = np.empty(self.shape)

        for i in range(len(times)):
            times[i] = self[i].start_time

        self = self.reshape(shape)
        times = times.reshape(shape)
        return times

    def __copy__(self):
        cpy = data_container(shape = self.shape)

        for i in range(self.size):
            cpy.flat[i] = copy.copy(self.flat[i])

        return cpy
