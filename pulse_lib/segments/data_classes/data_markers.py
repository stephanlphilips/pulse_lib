"""
data class for markers.
"""
from pulse_lib.segments.data_classes.data_generic import parent_data
from pulse_lib.segments.utility.rounding import iround

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

    def reset_time(self, time = None):
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

    def update_end_time(self, t):
        if t + self.start_time > self.end_time:
            self.end_time = t + self.start_time

    @property
    def total_time(self):
        '''
        get the total time of this segment.
        '''
        return self.end_time

    def get_vmin(self,sample_rate = 1e9):
        return 0

    def get_vmax(self,sample_rate = 1e9):
        return self.pulse_amplitude

    def integrate_waveform(self, sample_rate):
        """
        as markers are connected to mateched inputs, we do not need to compenstate, hence no interagration of waveforms is needed.
        """
        return 0

    def append(self, other):
        '''
        Append two segments to each other, where the other segment is places after the first segment. Time is the total time of the first segment.
        Args:
            other (marker_data) : other pulse data object to be appended

        ** what to do with start time argument?
        '''
        self.add_data(other, -1)

    def add_data(self, other, time=None):
        '''
        Add segment data to this data segment.
        The data is added after time. If time is None, then it is added after start_time of this segment.
        If time is -1, then it will be added after end_time
        Args:
            other (pulse_data) : other pulse data object to be appended
            time (float) : time to add the data.
        '''
        if time is None:
            time = self.start_time
        elif time == -1:
            time = self.end_time

        other_shifted = other._shift_all_time(time)
        self.my_marker_data += other_shifted.my_marker_data

        self.end_time = max(self.end_time, time + other.end_time)

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
        new_data.my_marker_data = self.my_marker_data + other.my_marker_data

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

    def _render(self, sample_rate, ref_channel_states, LO):
        '''
        make a full rendering of the waveform at a predermined sample rate.
        '''
        # express in Gs/s
        sample_rate = sample_rate*1e-9

        t_tot = self.total_time

        # get number of points that need to be rendered
        t_tot_pt = iround(t_tot * sample_rate) + 1

        my_sequence = np.zeros(t_tot_pt)

        for data_points in self.my_marker_data:
            start = iround(data_points.start * sample_rate)
            stop = iround(data_points.stop * sample_rate)

            my_sequence[start:stop] = 1*self.pulse_amplitude

        return my_sequence[:-1]

    def get_metadata(self, name):
        # TODO: add all pulses ??
        return {}

