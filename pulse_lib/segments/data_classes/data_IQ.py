"""
data class to store IQ based signals.
"""
import numpy as np
import copy


import pulse_lib.segments.utility.segments_c_func as seg_func
from pulse_lib.segments.utility.segments_c_func import py_calc_value_point_in_between, get_effective_point_number
from pulse_lib.segments.data_classes.data_generic import parent_data, data_container
from dataclasses import dataclass

class envelope_generator():
    """
    Object that handles envelope functions that can be used in spin qubit experiments.
    Key properties 
        * Makes sure average amplitude of the evelope is the one expressed
        * Executes some subsampling functions to give greater time resolution than the sample rate of the AWG.
        * Allows for plotting the FT of the envelope function.
    """
    def __init__(self, AM_envelope_function, PM_envelope_function=None):
        """
        define envelope funnctions.
        Args
            AM_envelope_functoin (lamba M) : function where M is the number of points where the evelopen should be rendered for.
        """
        self.AM_envelope_function = AM_envelope_function
        self.PM_envelope_function = PM_envelope_function

    def get_AM_envelope(self, delta_t, sample_rate=1e9):
        """
        Render the envelope for the given waveshape (in init).
        
        Args:
            delta_t (float) : time of the pulse (5.6 ns)
            sample_rate (float) : number of samples per second (e.g. 1GS/s)
        """
        n_points = delta_t/sample_rate*1e9 #times by default in ns
       
        if n_points < 1: #skip
            return np.asarray([0])

        envelope_extended = self.AM_envelope_function(int(n_points)*10)
        np.trapz(np.abs(envelope_extended), dx = 1/sample_rate)


@dataclass
class IQ_data_single:
    """
    structure to save relevant information about marker data.
    """
    start : float
    stop : float
    amplitude : float
    start_phase : float
    frequency : float
    envelope : envelope_generator


class IQ_data(parent_data):
    """
    class that manages the data used for generating IQ data
    """
    def __init__(self, LO):
        super().__init__()
        self.qubit_LO = LO
        self.MW_pulse_data = []
        self.global_phase = 0
        # just introduced for wait command (no pulse data stored)
        self.my_pulse_data = np.zeros([1,2], dtype=np.double)

    def add_MW_data(self, data):
        '''
        add data with all the pulse information
        Args:
            data (IQ_data_single) : data object for MW data.
        '''
        self.MW_pulse_data( data)

    def slice_time(self, start, end):
        '''
        slice time in IQ_data class
        Args:
            start (double) : new start time of the pulse
            end (double) : new end time of the pulse
        '''
        super().slice_time(start, end)

        super().__slice_sin_data(self.MW_pulse_data, start, end)
        super().__slice_sin_data(self.MOD_IQ_data, start, end)
        super().__slice_sin_data(self.numpy_IQ_data, start, end)

    def _shift_all_time(self, time_shift):
        '''
        Make a copy of all the data and shift all the time

        Args:
            time_shift (double) : shift the time
        Returns:
            data_copy_shifted (pulse_data) : copy of own data
        '''
        data_copy_shifted = super()._shift_all_time(time_shift)

        data_copy_shifted._shift_time_IQ_data_obj(data_copy_shifted.simple_IQ_data)
        data_copy_shifted._shift_time_IQ_data_obj(data_copy_shifted.MOD_IQ_data)
        data_copy_shifted._shift_time_IQ_data_obj(data_copy_shifted.numpy_IQ_data)

        return data_copy_shifted

    def _shift_all_phases(self, phase):
        """
        shift all phases present in this object.
        Args:
            phase (double) : the around of microwave phase you want to move around.
        """
        self.global_phase += phase
        self._shift_phase_IQ_data_obj(self.simple_IQ_data, phase)
        self._shift_phase_IQ_data_obj(self.MOD_IQ_data, phase)
        self._shift_phase_IQ_data_obj(self.numpy_IQ_data, phase)

    @staticmethod
    def _shift_phase_IQ_data_obj(data, phase):
        """
        shift phase in a data list of microwave IQ signals
        Args:
            phase (double) : the amound of microwave phase you want to move around.
        """
        for i in data_cpy:
            i['phase'] += phase


    @staticmethod
    def _shift_time_IQ_data_obj(data, time_shift):
        """
        shift time in a data list of microwave IQ signals
        Args:
            data (list<dict>) : data object, e.g. self.simple_IQ_data
            time_shift (double) : time to be shifted
        """
        for i in data_cpy:
            i['start_time'] += time_shift
            i['stop_time'] += time_shift

    def append(self, other, time):
        '''
        Append two segments to each other, where the other segment is places after the first segment. Time is the total time of the first segment.
        Args:
            other (pulse_data) : other pulse data object to be appended
            time (double/None) : length that the first segment should be.
        '''
        super().append(other, time)

        other_time_shifted = other._shift_all_time(time)
        other_time_shifted._shift_all_phases(self.global_phase)

        self.simple_IQ_data += other_time_shifted.simple_IQ_data   
        self.MOD_IQ_data += other_time_shifted.MOD_IQ_data   
        self.numpy_IQ_data += other_time_shifted.numpy_IQ_data   


    @property
    def total_time(self,):
        total_time = 0
        for IQ_data_item in self.simple_IQ_data:
            if IQ_data_item['stop_time'] > total_time:
                total_time = IQ_data_item['stop_time']

        for IQ_data_item in self.MOD_IQ_data:
            if IQ_data_item['stop_time'] > total_time:
                total_time = IQ_data_item['stop_time']

        for IQ_data_item in self.numpy_IQ_data:
            if IQ_data_item['stop_time'] > total_time:
                total_time = IQ_data_item['stop_time']

        if self.my_pulse_data[-1,0] > total_time:
            total_time = self.my_pulse_data[-1,0]

        return total_time

    def reset_time(self, time = None, extend_only = False):
        if extend_only == False:
            if time is not None:
                self.start_time = time
            else:
                self.start_time = self.total_time

        pulse = np.asarray([[0, 0],[self.start_time, 0]])
        self.add_pulse_data(pulse)

    def __copy__(self,):
        '''
        make a copy of self.
        '''
        my_copy = IQ_data(self.LO)
        my_copy.simple_IQ_data = copy.copy(self.simple_IQ_data)
        my_copy.MOD_IQ_data = copy.copy(self.MOD_IQ_data)
        my_copy.numpy_IQ_data = copy.copy(self.numpy_IQ_data)
        my_copy.global_phase = copy.copy(self.global_phase)
        my_copy.start_time = copy.copy(self.start_time)
        my_copy.my_pulse_data = copy.copy(self.my_pulse_data)
        return my_copy

    def get_IQ_data(self, I_or_Q):
        """
        get data object containing the I or Q part of the IQ signal
        Args:
            I_or_Q (str) : string 'I' or 'Q' to indicate which part of the signal to return
        Returns:
            new_data (pulse_data) : normal pulse_data object.
        """
        new_data = pulse_data()
        
        for i in self.simple_IQ_data:
            my_input = copy.copy(i)
            my_input['frequency'] -= self.LO
            if I_or_Q == 'Q':
                my_input['phase'] += np.pi/2
            new_data.add_sin_data(my_input)

        return new_data


    def __add__(self,):
        # aussume we do not need this.
        raise NotImplemented

    def __mul__(self):
        # aussume we do not need this.
        raise NotImplemented

    def _render(self, sample_rate, pre_delay = 0, post_delay = 0):
        '''
        make a full rendering of the waveform at a predermined sample rate. This should be defined in the child of this class.
        '''
        raise NotImplementedError("Pulse rendering not implemented for this type of pulse :-(")

    def get_vmax(self,sample_rate):
        raise NotImplemented
    
    def get_vmin(self,sample_rate):
        raise NotImplemented
    
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


