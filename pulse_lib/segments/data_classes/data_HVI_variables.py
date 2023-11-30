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
        self.data = dict()
        self.end_time = 0

    @property
    def HVI_markers(self):
        return self.data.copy()

    def __getitem__(self, *item):
        key = item[0]
        return self.data[key]

    def add_HVI_marker(self, name, value):
        """
        add a marker

        Args:
            name (str) : variable name for the HVI marker
            value (Any) : value to store
        """
        self.data[name] = value

    def reset_time(self, time=None):
        pass

    def wait(self, time):
        raise NotImplementedError()

    def integrate_waveform(self, sample_rate):
        raise NotImplementedError()

    def append(self, other):
        raise NotImplementedError()

    def add_data(self, other, time=None):
        raise NotImplementedError()

    def __copy__(self):
        """
        make a copy of this marker.
        """
        my_copy = marker_HVI_variable()
        my_copy.data = copy.copy(self.data)

        return my_copy

    def __add__(self, other):
        """
        add other maker to this one

        Args:
            other (marker_HVI_variable) : other marker object you want to add
        """

        if not isinstance(other, marker_HVI_variable):
            raise ValueError("only HVI makers can be added to HVI makers. No other types allowed.")

        new_data = marker_HVI_variable()
        new_data.data = {**self.data, **other.data}

        return new_data

    def __mul__(self, other):
        raise NotImplementedError()

    def __repr__(self):
        return (
            "=== raw data in HVI variable object ===\n\namplitude data ::\n"
            + str(self.my_amp_data) + "\ntime dep data ::\n" + str(self.my_time_data)
            )

    def _render(self, sample_rate, ref_channel_states, LO):
        '''
        make a full rendering of the waveform at a predetermined sample rate.
        '''
        raise ValueError("Rendering of HVI marker is currently not supported.")
