"""
data class to make pulses.
"""
import numpy as np
import copy
from dataclasses import dataclass
from typing import Any, Dict, Callable, List

import pulse_lib.segments.utility.segments_c_func as seg_func
from pulse_lib.segments.utility.segments_c_func import py_calc_value_point_in_between, get_effective_point_number
from pulse_lib.segments.data_classes.data_generic import parent_data, data_container
from pulse_lib.segments.data_classes.data_IQ import envelope_generator
from pulse_lib.segments.data_classes.data_pulse_core import pulse_data_single_sequence, base_pulse_element
# import time as tm

@dataclass
class custom_pulse_element:
    start: float
    stop: float
    func: Callable[..., np.ndarray]
    kwargs: Dict[str,Any]

def shift_start_stop(data:List[Any], delta) -> None:
    for element in data:
        element.start += delta
        element.stop += delta
    return data

def get_max_time(data:List[Any]) -> float:
    stop = 0
    for element in data:
        stop = max(stop, element.stop)
    return stop

class pulse_data(parent_data):
    """
    class defining base (utility) operations for baseband and microwave pulses.
    """
    def __init__(self):
        super().__init__()
        self.baseband_pulse_data = pulse_data_single_sequence()
        self.MW_pulse_data = list()
        self.custom_pulse_data = list()

        self.start_time = 0
        self.MW_end_time = 0
        self.global_phase = 0
        self.custom_end_time = 0

    def add_pulse_data(self, my_input):
        self.baseband_pulse_data.add_pulse(my_input)

    def add_MW_data(self, MW_data_object):
        """
        add object that defines a microwave pulse.

        Args:
            MW_data_object (IQ_data_single) : description MW pulse (see pulse_lib.segments.data_classes.data_IQ)
        """
        self.MW_pulse_data.append(MW_data_object)
        if self.MW_end_time < MW_data_object.stop:
            self.MW_end_time = MW_data_object.stop

    def add_custom_pulse_data(self, custom_pulse:custom_pulse_element):
        self.custom_pulse_data.append(custom_pulse)
        if self.custom_end_time < custom_pulse.stop:
            self.custom_end_time = custom_pulse.stop

    @property
    def total_time(self):
        '''
        total time of the current segment

        Returns:
            total_time (float) : total time of the segment.
        '''
        return max(self.baseband_pulse_data.total_time, self.MW_end_time, self.custom_end_time)

    def reset_time(self, time,  extend_only = False):
        '''
        Preform a reset time on the current segment.
        Args:
            time (float) : time where you want the reset. Of None, the totaltime of the segment will be taken.
            extend_only (bool) : will just extend the time in the segment and not reset it if set to true [do not use when composing wavoforms...].
        '''
        if time is None:
            time = self.total_time

        pulse = base_pulse_element(0,time,0,0)
        self.add_pulse_data(pulse)

        if extend_only == False:
            self.start_time = time

    def wait(self, time):
        """
        Wait after last point for x ns (note that this does not reset time)

        Args:
            time (double) : time in ns to wait
        """
        wait_time = self.total_time + time
        pulse = base_pulse_element(0,wait_time,0,0)

        self.add_pulse_data(pulse)

    def append(self, other, time = None):
        '''
        Append two segments to each other, where the other segment is places after the first segment. Time is the total time of the first segment.
        Args:
            other (pulse_data) : other pulse data object to be appended
            time (double/None) : length that the first segment should be.

        ** what to do with start time argument?
        '''
        if time is None:
            time = self.total_time
        else:
            self.slice_time(0, time)


        other_MW_pulse_data = copy.deepcopy(other.MW_pulse_data)
        shift_start_stop(other_MW_pulse_data, time)
        other_custom_pulse_data = copy.deepcopy(other.custom_pulse_data)
        shift_start_stop(other_custom_pulse_data, time)

        self.baseband_pulse_data.append(other.baseband_pulse_data)
        self.MW_pulse_data += other_MW_pulse_data
        self.custom_pulse_data += other_custom_pulse_data

        self.MW_end_time = get_max_time(self.MW_pulse_data)
        self.custom_end_time = get_max_time(self.custom_pulse_data)

    def repeat(self, n):
        """
        repeat n times
        Args
            n (int) : number of times to repeat
        """
        time = self.total_time
        new_MW_pulse_data =  copy.copy(self.MW_pulse_data)
        new_custom_pulse_data =  copy.copy(self.custom_pulse_data)

        for i in range(n):
            shifted_MW_pulse_data = copy.deepcopy(self.MW_pulse_data)
            shift_start_stop(shifted_MW_pulse_data, (i+1)*time)
            shifted_custom_pulse_data = copy.deepcopy(self.custom_pulse_data)
            shift_start_stop(shifted_custom_pulse_data, (i+1)*time)
            new_MW_pulse_data +=  shifted_MW_pulse_data
            new_custom_pulse_data +=  shifted_custom_pulse_data

        self.MW_pulse_data = new_MW_pulse_data
        self.custom_pulse_data = new_custom_pulse_data

        self.MW_end_time = get_max_time(self.MW_pulse_data)
        self.custom_end_time = get_max_time(self.custom_pulse_data)

        self.baseband_pulse_data.repeat(n)


    def slice_time(self, start, end):
        '''
        slice the pulse
        Args:
            Start (double) : enforced minimal starting time
            End (double) : enforced max time
        '''
        self.baseband_pulse_data.slice_time(start, end)
        self.__slice_MW_data(start, end)
        self.__slice_custom_pulse_data(start, end)

    '''
    Properties of the waveform
    '''

    def get_vmax(self,sample_rate = 1e9):
        '''
        calculate the maximum voltage in the current segment_single.

        If sine waves included, will take the maximum of the total render. If not, it just takes the max of the pulse data.
        '''
        if len(self.MW_pulse_data) == 0 and len(self.custom_pulse_data) == 0:
            return self.baseband_pulse_data.v_max
        else:
            return np.max(self.render(sample_rate=1e9))

    def get_vmin(self,sample_rate = 1e9):
        '''
        calculate the maximum voltage in the current segment_single.

        If sine waves included, will take the minimum of the total render. If not, it just takes the min of the pulse data.
        '''
        if len(self.MW_pulse_data) == 0 and len(self.custom_pulse_data) == 0:
            return self.baseband_pulse_data.v_min
        else:
            return np.min(self.render(sample_rate=1e9))

    def integrate_waveform(self, sample_rate):
        '''
        takes a full integral of the currently scheduled waveform.
        Args:
            sample_rate (double) : rate at which the AWG will be run
        Returns:
            integrate (double) : the integrated value of the waveform (unit is mV/sec).
        '''
        integrated_value = 0

        times, voltages = self.baseband_pulse_data.pulse_data
        baseband_pulse = np.empty([len(times), 2])
        baseband_pulse[:,0] = times
        baseband_pulse[:,1] = voltages

        for i in range(len(baseband_pulse)-1):
            integrated_value += (baseband_pulse[i,1] + baseband_pulse[i+1,1])/2*(baseband_pulse[i+1,0] - baseband_pulse[i,0])

        for custom_pulse in self.custom_pulse_data:
            integrated_value += np.sum(self._render_custom_pulse(custom_pulse, sample_rate))

        integrated_value *= 1e-9

        return integrated_value

    '''
    details of pulse data methods
    '''
    def __slice_MW_data(self, start, end):
        '''
        slice MW_data

        Args:
            Start (double) : enforced minimal starting time
            End (double) : enforced max time
        '''
        new_MW_data = []

        for i in self.MW_pulse_data:
            if i.start < start:
                i.start = start
            if i.stop > end:
                i.stop = end

            if i.start < i.stop:
                i.start -= start
                i.stop -= start
                new_MW_data.append(i)

        self.MW_pulse_data = new_MW_data

    def __slice_custom_pulse_data(self, start, end):
        '''
        slice custom_data

        Args:
            Start (double) : enforced minimal starting time
            End (double) : enforced max time
        '''
        new_custom_data = []

        for i in self.custom_pulse_data:
            if i.start < start:
                i.start = start
            if i.stop > end:
                i.stop = end

            if i.start < i.stop:
                i.start -= start
                i.stop -= start
                new_custom_data.append(i)

        self.custom_pulse_data = new_custom_data

    def shift_MW_frequency(self, frequency):
        '''
        shift the frequency of a MW signal that is defined. This is needed for dealing with the upconverion of a IQ signal.

        Args:
            frequency (float) : frequency you want to shift
        '''
        for IQ_data_single_object in self.MW_pulse_data:
            IQ_data_single_object.frequency -= frequency

    def shift_MW_phases(self, phase_shift):
        '''
        Shift the phases of all the microwaves present in the MW data object

        Args:
            phase_shift (float) : amount of phase to shift in rad.
        '''
        if phase_shift == 0:
            return

        for IQ_data_single_object in self.MW_pulse_data:
            IQ_data_single_object.start_phase += phase_shift

    def _shift_all_time_MW(self, time_shift):
        '''
        Shift time of all microwave pulses in memory.

        Args:
            time_shift (double) : shift the time
        '''

        for IQ_data_single_object in self.MW_pulse_data:
            IQ_data_single_object.start += time_shift
            IQ_data_single_object.stop += time_shift



    @staticmethod
    def _insert_pulse_arrays(src_array, to_insert, insert_position):
        '''
        insert pulse points in array
        Args:
            src_array : 2D pulse table
            to_insert : 2D pulse table to be inserted in the source
            insert_position: after which point the insertion needs to happen
        '''

        # calculate how long the piece is you want to insert
        dim_insert = len(to_insert)
        insert_position += 1

        new_arr = np.zeros([src_array.shape[0]+dim_insert, src_array.shape[1]])

        new_arr[:insert_position, :] = src_array[:insert_position, :]
        new_arr[insert_position:(insert_position + dim_insert), :] = to_insert
        new_arr[(insert_position + dim_insert):] = src_array[insert_position :]

        return new_arr

    '''
    operators for the data object.
    '''
    def __copy__(self):
        my_copy = pulse_data()
        my_copy.baseband_pulse_data = copy.copy(self.baseband_pulse_data)
        my_copy.MW_pulse_data = copy.deepcopy(self.MW_pulse_data)
        my_copy.custom_pulse_data = copy.deepcopy(self.custom_pulse_data)
        my_copy.start_time = copy.copy(self.start_time)
        my_copy.software_marker_data = copy.copy(self.software_marker_data)
        my_copy.global_phase = copy.copy(self.global_phase)
        my_copy.MW_end_time = get_max_time(my_copy.MW_pulse_data)
        my_copy.custom_end_time = get_max_time(my_copy.custom_pulse_data)

        return my_copy

    def __add__(self, other):
        '''
        define addition operator for pulse_data object
        '''
        new_data = pulse_data()
        if type(other) is pulse_data:
            # is there a need for copy command  -- investigate is this would start effecting performance.
            new_data.baseband_pulse_data = copy.copy(self.baseband_pulse_data)
            new_data.baseband_pulse_data += other.baseband_pulse_data
            new_data.MW_pulse_data = self.MW_pulse_data + other.MW_pulse_data
            new_data.custom_pulse_data = self.custom_pulse_data + other.custom_pulse_data

        elif type(other) == int or type(other) == float:
            new_pulse = copy.copy(self.baseband_pulse_data)
            new_pulse.add_pulse(base_pulse_element(0,-1, other, other))
            new_data.baseband_pulse_data = new_pulse

            new_data.MW_pulse_data = copy.copy(self.MW_pulse_data)
            new_data.custom_pulse_data = copy.copy(self.custom_pulse_data)

        else:
            raise TypeError("Please add up pulse_data object (or pulse/IQ segment type) type or a number ")

        return new_data

    def __mul__(self, other):
        '''
        muliplication operator for segment_single
        '''
        new_data = pulse_data()

        if type(other) is pulse_data:
            raise NotImplemented
        elif type(other) == int or type(other) == float or type(other) == np.float64:
            new_data.baseband_pulse_data = copy.copy(self.baseband_pulse_data)
            new_data.baseband_pulse_data *= other

            for IQ_data_single_object in self.MW_pulse_data:
                IQ_data_single_object_cpy = copy.copy(IQ_data_single_object)
                IQ_data_single_object_cpy.amplitude *=other
                new_data.MW_pulse_data.append(IQ_data_single_object_cpy)

            new_data.custom_pulse_data = copy.copy(self.custom_pulse_data)

        else:
            raise TypeError("multiplication should be done with a number, type {} not supported".format(type(other)))

        return new_data

    def _render_custom_pulse(self, custom_pulse, sample_rate):
        duration = custom_pulse.stop - custom_pulse.start
        data = custom_pulse.func(duration, sample_rate, **custom_pulse.kwargs)
        return data

    def _render(self, sample_rate):
        '''
        make a full rendering of the waveform at a predetermined sample rate.
        '''

        # express in Gs/s
        sample_rate = sample_rate*1e-9
        sample_time_step = 1/sample_rate

        t_tot = self.total_time

        # get number of points that need to be rendered
        t_tot_pt = get_effective_point_number(t_tot, sample_time_step) + 1

        my_sequence = np.zeros([int(t_tot_pt)])
        # start rendering pulse data

        # TODO upgrade to new format -- put in the cython part for better performance..
        times, voltages = self.baseband_pulse_data.pulse_data
        baseband_pulse = np.empty([len(times), 2])
        baseband_pulse[:,0] = times
        baseband_pulse[:,1] = voltages

        for i in range(0,len(baseband_pulse)-1):
            t0_pt = get_effective_point_number(baseband_pulse[i,0], sample_time_step)
            t1_pt = get_effective_point_number(baseband_pulse[i+1,0], sample_time_step) + 1
            t0 = t0_pt*sample_time_step
            t1 = t1_pt*sample_time_step
            if t0 > t_tot:
                continue
            elif t1 > t_tot + sample_time_step:
                if baseband_pulse[i,1] == baseband_pulse[i+1,1]:
                    my_sequence[t0_pt: t_tot_pt] = baseband_pulse[i,1]
                else:
                    val = py_calc_value_point_in_between(baseband_pulse[i,:], baseband_pulse[i+1,:], t_tot)
                    my_sequence[t0_pt: t_tot_pt] = np.linspace(
                        baseband_pulse[i,1],
                        val, t_tot_pt-t0_pt)
            else:
                if baseband_pulse[i,1] == baseband_pulse[i+1,1]:
                    my_sequence[t0_pt: t1_pt] = baseband_pulse[i,1]
                else:
                    my_sequence[t0_pt: t1_pt] = np.linspace(baseband_pulse[i,1], baseband_pulse[i+1,1], t1_pt-t0_pt)
        # top off the sequence -- default behavior, extend the last value
        if len(baseband_pulse) > 1:
            pt = get_effective_point_number(baseband_pulse[i+1,0], sample_time_step)
            my_sequence[pt:] = baseband_pulse[i+1,1]

        # render MW pulses.
        for IQ_data_single_object in self.MW_pulse_data:
            # start stop time of MW pulse

            start_pulse = IQ_data_single_object.start
            stop_pulse = IQ_data_single_object.stop

            # max amp, freq and phase.
            amp  =  IQ_data_single_object.amplitude
            freq =  IQ_data_single_object.frequency
            phase = IQ_data_single_object.start_phase

            # evelope data of the pulse
            if IQ_data_single_object.envelope is None:
                IQ_data_single_object.envelope = envelope_generator()

            amp_envelope = IQ_data_single_object.envelope.get_AM_envelope((stop_pulse - start_pulse), sample_rate)
            phase_envelope = IQ_data_single_object.envelope.get_PM_envelope((stop_pulse - start_pulse), sample_rate)

            #self.baseband_pulse_data[-1,0] convert to point numbers
            n_pt = len(amp_envelope)
            start_pt = get_effective_point_number(start_pulse, sample_time_step)
            stop_pt = start_pt + n_pt

            # add the sin pulse
            my_sequence[start_pt:stop_pt] += amp*amp_envelope*np.sin(
                    2*np.pi*freq/sample_rate*1e-9*(np.arange(n_pt)+start_pt)
                    + phase + phase_envelope )

        for custom_pulse in self.custom_pulse_data:
            data = self._render_custom_pulse(custom_pulse, sample_rate*1e9)
            start_pt = get_effective_point_number(custom_pulse.start, sample_time_step)
            stop_pt = start_pt + len(data)
            my_sequence[start_pt:stop_pt] += data

        # remove last value. t_tot_pt = t_tot + 1. Last value is always 0. It is only needed in the loop on the baseband pulses.
        return my_sequence[:-1]

if __name__ == '__main__':
    """
    test functions for the IQ_data object
    """
    from pulse_lib.segments.data_classes.data_IQ import IQ_data_single
    import matplotlib.pyplot as plt
    # make two shaped pulses after each other.
    IQ_data_object1 = IQ_data_single()
    IQ_data_object1.start = 10
    IQ_data_object1.stop = 140
    IQ_data_object1.amplitude = 1
    IQ_data_object1.frequency = 1e8
    IQ_data_object1.start_phase = np.pi/2
    IQ_data_object1.envelope = envelope_generator('flattop')

    IQ_data_object2 = IQ_data_single()
    IQ_data_object2.start = 150
    IQ_data_object2.stop = 190
    IQ_data_object2.amplitude = 1
    IQ_data_object2.frequency = 1e8

    data = pulse_data()
    data.add_MW_data(IQ_data_object1)
    data.reset_time(data.total_time)
    data.add_MW_data(IQ_data_object2)
    data.reset_time(data.total_time)
    data.add_pulse_data(base_pulse_element(0,140, 1 , 1))
    data.add_pulse_data(base_pulse_element(200,280, 2 , 1))

    data.repeat(2)
    rendering_data = data.render(1e9)
    t = np.linspace(0, data.total_time, len(rendering_data))
    plt.plot(t, rendering_data)
    plt.show()