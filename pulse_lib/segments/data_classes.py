import numpy as np
import copy


import pulse_lib.segments.segments_c_func as seg_func
from pulse_lib.segments.segments_c_func import py_calc_value_point_in_between

import time

class pulse_data():
    """object that saves all the pulse data that is present in an segment object.
    This object support all the fundametal operations needed to define the segments."""
    def __init__(self):
        self.my_pulse_data = np.zeros([1,2], dtype=np.double)
        self.sin_data = []
        self.sim_mod_data = []

        self.numpy_data = []

        self.start_time = 0
        self.waveform_cache = dict() 

    def add_pulse_data(self, input):
        self.my_pulse_data = self._add_up_pulse_data(input)

    def add_sin_data(self, input):
        self.sin_data.append(input)

    def add_numpy_data(self, input):
        raise NotImplemented

    def slice_time(self, start, end):
        '''
        slice the time in the pulse data object.
        Args:
            start (double) : new starting time of the pulse (>= 0)
            end (double) : new end time of the pulse.
        Note that the start and end point of the slice will correspond to the zero base level of the pulse.
        (you can set the base level by doing my_segment += <double> base_level)
        '''
        if start < 0 :
            raise ValueError("Error slicing pulse, start time of a pulse cannot be smaller than 0!")

        # make sure we have data points of start and end in my_pulse_data:
        fake_pulse = np.zeros([2,2], np.double)
        fake_pulse[0,0] = start
        fake_pulse[1,0] = end

        self.my_pulse_data = self._add_up_pulse_data(fake_pulse)

        start_idx = np.where(self.my_pulse_data[:,0] == start)[0][0]
        end_idx = np.where(self.my_pulse_data[:,0] == end)[0][0] +1

        self.my_pulse_data = self.my_pulse_data[start_idx:end_idx]
        self.my_pulse_data[:,0] -= start

        self.__slice_sin_data(self.sin_data, start, end)

    @staticmethod
    def __slice_sin_data(sin_data, start, end):
        '''
        slice sin_data
        Args:
            sin_data (list<dict>) : object that contains a dict describing a sinus-based pulse
            start (double) : enforced minimal starting time
            end (double) : enforced max time
        '''

        new_sin_data = []

        for i in sin_data:
            if i['start_time'] < start:
                i['start_time'] = start
            if i['stop_time'] > end:
                i['stop_time'] = end

            if i['start_time'] < i['stop_time']:
                i['start_time'] -= start
                i['stop_time'] -= start
                new_sin_data.append(i)

        sin_data = new_sin_data

    def _shift_all_time(self, time_shift):
        '''
        Make a copy of all the data and shift all the time

        Args:
            time_shift (double) : shift the time
        Returns:
            data_copy_shifted (pulse_data) : copy of own data
        '''
        if time_shift <0 :
            raise ValueError("when shifting time, you cannot make negative times. Apply a positive shift.")
        data_copy_shifted = copy.copy(self)
        data_copy_shifted.my_pulse_data[:,0] += time_shift

        for i in self.sin_data:
            i['start_time'] += time_shift
            i['stop_time'] += time_shift

        return data_copy_shifted
    @property
    def total_time(self,):
        total_time = 0

        for sin_data_item in self.sin_data:
            if sin_data_item['stop_time'] > total_time:
                total_time = sin_data_item['stop_time']

        if self.my_pulse_data[-1,0] > total_time:
            total_time = self.my_pulse_data[-1,0]
        return total_time

    def reset_time(self, time = None, extend_only = False):

        if time is not None:
            pulse = np.asarray([[0, 0],[time, 0]])
            self.add_pulse_data(pulse)
        if extend_only == False:
            self.start_time = self.total_time
        
    def get_vmax(self,sample_rate = 1e9):
        '''
        calculate the maximum voltage in the current segment_single.

        If sine waves included, will take the maximum of the total render. Tf not, it just takes the max of the pulse data.
        '''
        if len(self.sin_data) == 0:
            return np.max(self.my_pulse_data[:,1])
        else:
            return np.max(self.render(sample_rate=1e9, clear_cache_on_exit = False))

    def get_vmin(self,sample_rate = 1e9):
        '''
        calculate the maximum voltage in the current segment_single.

        If sine waves included, will take the minimum of the total render. Tf not, it just takes the min of the pulse data.
        ''' 
        if len(self.sin_data) == 0:
            return np.min(self.my_pulse_data[:,1])
        else:   
            return np.min(self.render(sample_rate=1e9, clear_cache_on_exit = False))

    def integrate_waveform(self, pre_delay, post_delay, sample_rate):
        '''
        takes a full integral of the currently scheduled waveform.
        Args:
            start_time (double) : from which points the rendering needs to start
            stop_time (double) : to which point the rendering needs to go (default (-1), to entire segement)
            sample_rate (double) : rate at which the AWG will be run
        Returns:
            integrate (double) : the integrated value of the waveform (unit is mV/sec).
        '''
        integrated_value = 0
        # if len(self.sin_data) == 0 and pre_delay <= 0 and post_delay>= 0:

        sample_rate = self.waveform_cache['sample_rate']*1e-9
        sample_time_step = 1/sample_rate
        pre_delay_eff = get_effective_point_number(pre_delay, sample_time_step)*sample_time_step
        post_delay_eff = get_effective_point_number(post_delay, sample_time_step)*sample_time_step

        for i in range(len(self.my_pulse_data)-1):
            integrated_value += (self.my_pulse_data[i,1] + self.my_pulse_data[i+1,1])/2*(self.my_pulse_data[i+1,0] - self.my_pulse_data[i,0])
        integrated_value += pre_delay_eff*self.my_pulse_data[0,1] +post_delay_eff*self.my_pulse_data[-1,1]
        integrated_value *= 1e-9
        # else: # slow way ...
        #     wvf = self.render(pre_delay, post_delay, sample_rate, clear_cache_on_exit = False)
        #     integrated_value = np.trapz(wvf, dx=1/sample_rate)

        return integrated_value

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
                'waveform' : self.__render(sample_rate, pre_delay_wvf, post_delay_wvf),
                'pre_delay': pre_delay,
                'post_delay' : post_delay
            }

        # get the waveform
        my_waveform = self.get_resized_waveform(pre_delay, post_delay)
        
        # clear cache if needed
        if clear_cache_on_exit == True:
            self.waveform_cache = dict()

        return my_waveform

    def __render(self, sample_rate, pre_delay = 0, post_delay = 0):
        '''
        make a full rendering of the waveform at a predermined sample rate.
        '''
        # express in Gs/s
        sample_rate = sample_rate*1e-9
        sample_time_step = 1/sample_rate
                
        t_tot = self.total_time

        # get number of points that need to be rendered
        t_tot_pt = get_effective_point_number(t_tot, sample_time_step) + 1
        pre_delay_pt = - get_effective_point_number(pre_delay, sample_time_step)
        post_delay_pt = get_effective_point_number(post_delay, sample_time_step)

        my_sequence = np.zeros([int(t_tot_pt + pre_delay_pt + post_delay_pt)])
        # start rendering pulse data
        for i in range(0,len(self.my_pulse_data)-1):
            t0_pt = get_effective_point_number(self.my_pulse_data[i,0], sample_time_step)
            t1_pt = get_effective_point_number(self.my_pulse_data[i+1,0], sample_time_step) + 1
            t0 = t0_pt*sample_time_step
            t1 = t1_pt*sample_time_step
            if t0 > t_tot:
                continue
            elif t1 > t_tot + sample_time_step:
                if self.my_pulse_data[i,1] == self.my_pulse_data[i+1,1]:
                    my_sequence[t0_pt + pre_delay_pt: t_tot_pt + pre_delay_pt] = self.my_pulse_data[i,1]
                else:
                    val = py_calc_value_point_in_between(self.my_pulse_data[i,:], self.my_pulse_data[i+1,:], t_tot)
                    my_sequence[t0_pt + pre_delay_pt: t_tot_pt + pre_delay_pt] = np.linspace(
                        self.my_pulse_data[i,1], 
                        val, t_tot_pt-t0_pt)
            else:
                if self.my_pulse_data[i,1] == self.my_pulse_data[i+1,1]:
                    my_sequence[t0_pt + pre_delay_pt: t1_pt + pre_delay_pt] = self.my_pulse_data[i,1]
                else:
                    my_sequence[t0_pt + pre_delay_pt: t1_pt + pre_delay_pt] = np.linspace(self.my_pulse_data[i,1], self.my_pulse_data[i+1,1], t1_pt-t0_pt)
        # top off the sequence -- default behavior, extend the last value
        if len(self.my_pulse_data) > 1:
            pt = get_effective_point_number(self.my_pulse_data[i+1,0], sample_time_step)
            my_sequence[pt + pre_delay_pt:] = self.my_pulse_data[i+1,1]

        for sin_data_item in self.sin_data:
            if sin_data_item['start_time'] > t_tot:
                continue
            elif sin_data_item['stop_time'] > t_tot:
                stop = t_tot_pt + pre_delay_pt
            else:
                stop =  get_effective_point_number(sin_data_item['stop_time'], sample_time_step) + pre_delay_pt
            
            start = get_effective_point_number(sin_data_item['start_time'], sample_time_step) + pre_delay_pt
            start_t  = (start - pre_delay_pt)*sample_time_step
            stop_t  = (stop - pre_delay_pt)*sample_time_step

            if sin_data_item['type'] == 'std':
                amp  =  sin_data_item['amplitude']
                freq =  sin_data_item['frequency']
                phase = sin_data_item['phase']
                print(amp, freq, phase)
                my_sequence[start:stop] += amp*np.sin(np.linspace(start_t, stop_t-sample_time_step, stop-start)*freq*1e-9*2*np.pi + phase)
            else: 
                raise ValueError("type of sin pulse not implemented. currently only standard pulses supported")

        return my_sequence      

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
                new_waveform[0:n_pt_before] = self.my_pulse_data[0,1]
                if n_pt_after < 0:
                    new_waveform[n_pt_before:] = self.waveform_cache['waveform'][:n_pt_after]
                elif n_pt_after == 0:
                    new_waveform[n_pt_before:] = self.waveform_cache['waveform']
                else:
                    new_waveform[n_pt_before:-n_pt_after] = self.waveform_cache['waveform']
            else:
                new_waveform[:-n_pt_after] = self.waveform_cache['waveform'][-n_pt_before:]

            if n_pt_after > 0:
                new_waveform[-n_pt_after:] =  self.my_pulse_data[-1,1]

            return new_waveform

    def _add_up_pulse_data(self, new_pulse):
        '''
        add a pulse up to the current pulse in the memory.
        new_pulse --> default format as in the add_pulse function
        '''
        my_pulse_data_copy = self.my_pulse_data
        # step 1: make sure both pulses have the same length
        if self.total_time < new_pulse[-1,0]:
            to_insert = [[new_pulse[-1,0],my_pulse_data_copy[-1,1]]]
            my_pulse_data_copy = self._insert_arrays(my_pulse_data_copy, to_insert, len(my_pulse_data_copy)-1)
        elif self.total_time > new_pulse[-1,0]:
            to_insert = [[my_pulse_data_copy[-1,0],new_pulse[-1,1]]]
            new_pulse = self._insert_arrays(new_pulse, to_insert, len(new_pulse)-1)
            
        my_pulse_data_tmp, new_pulse_tmp = seg_func.interpolate_pulses(my_pulse_data_copy, new_pulse)

        final_pulse = np.zeros([len(my_pulse_data_tmp),2])
        final_pulse[:,0] = my_pulse_data_tmp[:,0]
        final_pulse[:,1] +=  my_pulse_data_tmp[:,1]  + new_pulse_tmp[:,1]

        return final_pulse

    @staticmethod
    def _insert_arrays(src_array, to_insert, insert_position):
        '''
        insert pulse points in array
        Args:
            src_array : 2D pulse table
            to_insert : 2D pulse table to be inserted in the source
            insert_position: after which point the insertion needs to happen
        '''

        # calcute how long the piece is you want to insert
        dim_insert = len(to_insert)
        insert_position += 1

        new_arr = np.zeros([src_array.shape[0]+dim_insert, src_array.shape[1]])
        
        new_arr[:insert_position, :] = src_array[:insert_position, :]
        new_arr[insert_position:(insert_position + dim_insert), :] = to_insert
        new_arr[(insert_position + dim_insert):] = src_array[insert_position :]

        return new_arr

    def __copy__(self):
        my_copy = pulse_data()
        my_copy.my_pulse_data = copy.copy(self.my_pulse_data)
        my_copy.sin_data = copy.copy(self.sin_data)
        my_copy.numpy_data = copy.copy(self.numpy_data)
        my_copy.start_time = copy.copy(self.start_time)
        return my_copy

    def __add__(self, other):
        '''
        define addition operator for segment_single
        '''
        new_data = pulse_data()
        if type(other) is pulse_data:
            if len(other.my_pulse_data) == 1:
                new_data.my_pulse_data = copy.copy(self.my_pulse_data)
            elif len(self.my_pulse_data) == 1:
                new_data.my_pulse_data = copy.copy(other.my_pulse_data)
            else:
                new_data.my_pulse_data = self._add_up_pulse_data(other.my_pulse_data)

            sin_data = copy.copy(self.sin_data)
            sin_data.extend(other.sin_data)
            new_data.sin_data = sin_data
        elif type(other) == int or type(other) == float:
            new_pulse = copy.copy(self.my_pulse_data)
            new_pulse[:,1] += other
            new_data.my_pulse_data = new_pulse
            new_data.sin_data = self.sin_data

        else:
            raise TypeError("Please add up segment_single type or a number ")

        return new_data

    def __mul__(self, other):
        '''
        muliplication operator for segment_single
        '''
        new_data = pulse_data()

        if type(other) is pulse_data:
            raise NotImplemented
        elif type(other) == int or type(other) == float or type(other) == np.float64:
            new_pulse = copy.copy(self.my_pulse_data)
            new_pulse[:,1] *= other
            new_data.my_pulse_data = new_pulse
            sin_data = []
            for i in self.sin_data:
                my_sin = copy.copy(i)
                my_sin['amplitude'] *=other
                sin_data.append(my_sin)

            new_data.sin_data = sin_data
        else:
            raise TypeError("muliplication shoulf be done with a number, type {} not supported".format(type(other)))
        
        return new_data

    def append(self, other, time):
        '''
        Append two segments to each other, where the other segment is places after the first segment. Time is the total time of the first segment.
        Args:
            other (pulse_data) : other pulse data object to be appended
            time (double/None) : length that the first segment should be.

        ** what to do with start time argument?
        '''

        self.slice_time(0, time)

        other_time_shifted = other._shift_all_time(time)
        new_sin_data =  self.sin_data +  other_time_shifted.sin_data

        len_pulse_a = len(self.my_pulse_data)
        if len_pulse_a > 2 and self.my_pulse_data[-1,0] == self.my_pulse_data[-2,0]:
                len_pulse_a -= 1
        len_pulse_b = len(other_time_shifted.my_pulse_data)
        if len_pulse_b > 2 and other_time_shifted.my_pulse_data[0,0] == other_time_shifted.my_pulse_data[1,0]:
                len_pulse_b -= 1

        new_pulse_data = np.zeros([len_pulse_a + len_pulse_b,2], dtype=np.double)
        new_pulse_data[:len_pulse_a] = self.my_pulse_data[:len_pulse_a]
        new_pulse_data[len_pulse_a:] = other_time_shifted.my_pulse_data[-len_pulse_b:]

        self.my_pulse_data = new_pulse_data
        self.sin_data = new_sin_data

class IQ_data(pulse_data):
    """
    class that manages the data used for generating IQ data
    """
    def __init__(self, LO):
        self.LO = LO
        self.simple_IQ_data = []
        self.MOD_IQ_data = []
        self.numpy_IQ_data = []
        self.start_time = 0
        self.global_phase = 0
        # just introduced for wait command (no pulse data stored)
        self.my_pulse_data = np.zeros([1,2], dtype=np.double)

    def add_simple_data(self, input_dict):
        self.simple_IQ_data.append(input_dict)
    
    def add_mod_data (self, input_dict):
        self.simple_IQ_data.append(input_dict)

    def add_numpy_IQ(self, input_dict):
        self.numpy_IQ_data.append(input_dict)

    def slice_time(self, start, end):
        '''
        slice time in IQ_data class
        Args:
            start (double) : new start time of the pulse
            end (duoble) : new end time of the pulse
        '''
        super().slice_time(start, end)

        super().__slice_sin_data(self.simple_IQ_data, start, end)
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
            phase (double) : the amound of microwave phase you want to move around.
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

def get_effective_point_number(time, time_step):
    '''
    function that discretizes time depending on the sample rate of the AWG.
    Args:
        time (double): time in ns of which you want to know how many points the AWG needs to get there
        time_step (double) : time step of the AWG (ns)

    Returns:
        how many points you need to get to the desired time step.
    '''
    n_pt, mod = divmod(time, time_step)
    if mod > time_step/2:
        n_pt += 1

    return int(n_pt)