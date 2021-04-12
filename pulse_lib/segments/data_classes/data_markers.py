"""
data class for markers.
"""
from pulse_lib.segments.data_classes.data_generic import parent_data
from pulse_lib.segments.utility.segments_c_func import get_effective_point_number

import numpy as np
import copy
from dataclasses import dataclass

@dataclass
class marker_pulse:
    start: float
    stop: float

class marker_data(parent_data):
    def __init__(self, pulse_amplitude = 1000):
        """
        init marker object
        Args:
            pulse_amplitude(double) : pulse amplitude in mV
        """
        super().__init__()
        self.my_marker_data =  list()

        self.start_time = 0
        self.end_time = 0

        self.pulse_amplitude = pulse_amplitude

    def add_marker(self, start, stop):
        """
        add a marker
        Args:
            start (double) : start time of the marker
            stop (double) : stop time of the marker
        """
        if stop < start:
            raise ValueError(f"Start time ({start}) should be > stop time ({stop})")
        self.my_marker_data.append( marker_pulse(start + self.start_time, stop + self.start_time) )

        if stop + self.start_time > self.end_time:
            self.end_time = self.start_time + stop

    def reset_time(self, time = None, extend_only = False):
        """
        reset the effective start time. See online manual in pulse building instructions to understand this command.
        Args:
            time (double) : new time that will become time zero
        """
        self.start_time = self.total_time
        if time is not None:
            self.start_time =time

        if self.start_time > self.end_time:
            self.end_time = self.start_time

    def wait(self, time):
        """
        Wait after marker for x ns.

        Args:
            time (double) : time in ns to wait
        """
        self.end_time += time

    @property
    def total_time(self):
        '''
        get the total time of this segment.
        '''
        return self.end_time

    def slice_time(self, start, end):
        """
        apply slice operation on this marker.
        Args:
            start (double) : start time of the marker
            stop (double) : stop time of the marker
        """
        sliced_maker_data = []
        for data_item in self.my_marker_data:
            in_range, data_item = slice_out_marker_single(start, end, data_item)
            if in_range == True:
                sliced_maker_data.append(data_item)


        self.my_marker_data = sliced_maker_data
        self.end_time = end - start

    def get_vmin(self,sample_rate = 1e9):
        return 0

    def get_vmax(self,sample_rate = 1e9):
        return self.pulse_amplitude

    def integrate_waveform(self, sample_rate):
        """
        as markers are connected to mateched inputs, we do not need to compenstate, hence no interagration of waveforms is needed.
        """
        return 0

    def append(self, other, time = None):
        '''
        Append two segments to each other, where the other segment is places after the first segment. Time is the total time of the first segment.
        Args:
            other (marker_data) : other pulse data object to be appended
            time (double/None) : length that the first segment should be.

        ** what to do with start time argument?
        '''
        end_time = self.end_time
        if time is not None:
            end_time = time
            self.slice_time(0, end_time)

        other_shifted = other._shift_all_time(end_time)
        self.my_marker_data += other_shifted.my_marker_data
        self.end_time += other.end_time

    def __copy__(self):
        """
        make a copy of this marker.
        """
        my_copy = marker_data(self.pulse_amplitude)
        my_copy.my_marker_data = copy.copy(self.my_marker_data)
        my_copy.start_time = copy.copy(self.start_time)
        my_copy.end_time = copy.copy(self.end_time)


        return my_copy

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

        for i in range(len(data_copy_shifted.my_marker_data)):
            data_copy_shifted.my_marker_data[i] = marker_pulse(data_copy_shifted.my_marker_data[i].start + time_shift,
                                                               data_copy_shifted.my_marker_data[i].stop + time_shift)


        return data_copy_shifted

    def __add__(self, other):
        """
        add other maker to this one
        Args:
            other (marker_data) : other marker object you want to add
        """

        if not isinstance(other, marker_data):
            raise ValueError("only markers can be added to markers. No other types allowed.")

        new_data = marker_data()
        new_data.my_marker_data += self.my_marker_data
        new_data.my_marker_data += other.my_marker_data

        new_data.start_time = self.start_time
        new_data.end_time= self.end_time

        if other.end_time > self.end_time:
            new_data.end_time = other.end_time

        return new_data

    def __mul__(self, other):
        raise ValueError("No multiplication support for markers ...")

    def print_all(self):
        for i in self.my_marker_data:
            print(i)

    def _render(self, sample_rate, ref_channel_states):
        '''
        make a full rendering of the waveform at a predermined sample rate.
        '''
        # express in Gs/s
        sample_rate = sample_rate*1e-9
        sample_time_step = 1/sample_rate

        t_tot = self.total_time

        # get number of points that need to be rendered
        t_tot_pt = get_effective_point_number(t_tot, sample_time_step) + 1

        my_sequence = np.zeros([int(t_tot_pt)])

        for data_points in self.my_marker_data:
            start = get_effective_point_number(data_points.start, sample_time_step)
            stop = get_effective_point_number(data_points.stop, sample_time_step)

            my_sequence[start:stop] = 1*self.pulse_amplitude

        return my_sequence[:-1]

def slice_out_marker_single(start, stop, start_stop_pulse):
    """
    check if start stop falls in valid range.
    Args:
        start (double) : startpoint of where the marker must be in
        end (double) : endpoint where the marker must be in.
        start_stop_position (marker_pulse) : tuple iwht start and stop point of the marker.
    Return:
        True/False if start and stop are not in range
        start_stop_position (tuple) : sliced time.

    Function also fixes the time in the pointer that is given.
    """
    if start_stop_pulse.stop <= start or start_stop_pulse.start >= stop:
        return False

    result = copy.copy(start_stop_pulse)
    if result.start < start:
        result.start = start

    if result.stop > stop:
        result.stop = stop

    result.start -= start
    result.stop -= start

    return True, result
