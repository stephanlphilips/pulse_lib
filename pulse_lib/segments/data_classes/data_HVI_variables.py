"""
data class for markers.
"""
from pulse_lib.segments.data_classes.data_generic import parent_data

import copy

class marker_HVI_variable(parent_data):
    def __init__(self):
        """
        init marker object

        Args:
            pulse_amplitude(double) : pulse amplitude in mV
        """
        super().__init__()
        self.my_time_data = dict()
        self.my_amp_data = dict()

        self.end_time = 0

    @property
    def HVI_markers(self):
        return {**self.my_time_data, **self.my_amp_data}

    def __getitem__(self, *item):
        key = item[0]
        try:
            return self.my_time_data[key]
        except:
            pass
        try:
            return self.my_amp_data[key]
        except:
            pass

        raise ValueError(f"Asking for HVI variable {key}. But this variable is not present in the current data set.")

    def add_HVI_marker(self, name, amplitude, time):
        """
        add a marker

        Args:
            name (str) : variable name for the HVI marker
            amplitude (float) : amplitude of the marker (in case of a time, unit is in ns, else mV)
            time (bool) : True is marker needs to be interpreted as a time.
        """
        if time == True:
            self.my_time_data[name] = amplitude
        else:
            self.my_amp_data[name] =  amplitude

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

    @property
    def total_time(self):
        '''
        get the total time of this segment.
        '''
        return self.end_time

    def get_vmin(self,sample_rate = 1e9):
        return 0

    def get_vmax(self,sample_rate = 1e9):
        return 0

    def integrate_waveform(self, sample_rate):
        """
        as markers are connected to matched inputs, we do not need to compensate, hence no integration of waveforms is needed.
        """
        return 0

    def append(self, other):
        '''
        Append two segments to each other, where the other segment is places after the first segment. Time is the total time of the first segment.

        Args:
            other (marker_HVI_variable) : other pulse data object to be appended
        '''
        end_time = self.total_time

        other_shifted = other._shift_all_time(end_time)
        self.my_time_data.update(other_shifted.my_time_data)
        self.my_amp_data.update(other.my_amp_data)

    def __copy__(self):
        """
        make a copy of this marker.
        """
        my_copy = marker_HVI_variable()
        my_copy.my_amp_data = copy.copy(self.my_amp_data)
        my_copy.my_time_data = copy.copy(self.my_time_data)
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

        for key in data_copy_shifted.my_time_data.keys():
            data_copy_shifted.my_time_data[key] += time_shift

        return data_copy_shifted

    def __add__(self, other):
        """
        add other maker to this one

        Args:
            other (marker_HVI_variable) : other marker object you want to add
        """

        if not isinstance(other, marker_HVI_variable):
            raise ValueError("only HVI makers can be added to HVI makers. No other types allowed.")

        new_data = marker_HVI_variable()
        new_data.my_time_data = {**self.my_time_data, **other.my_time_data}
        new_data.my_amp_data = {**self.my_amp_data, **other.my_amp_data}

        new_data.start_time = self.start_time
        new_data.end_time = self.end_time
        if other.total_time > self.total_time:
            new_data.end_time = other.end_time

        return new_data

    def __mul__(self, other):
        raise ValueError("No multiplication support for markers ...")

    def __repr__(self):
        return "=== raw data in HVI variable object ===\n\namplitude data ::\n" + str(self.my_amp_data) + "\ntime dep data ::\n" + str(self.my_time_data)

    def _render(self, sample_rate, ref_channel_states, LO):
        '''
        make a full rendering of the waveform at a predetermined sample rate.
        '''
        raise ValueError("Rendering of HVI marker is currently not supported.")

