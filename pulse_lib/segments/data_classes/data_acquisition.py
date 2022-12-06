"""
data class for acquisitions.
"""
import numpy as np
import copy
from dataclasses import dataclass
from typing import Optional

from pulse_lib.segments.data_classes.data_generic import parent_data


@dataclass
class acquisition:
    ref : Optional[str]
    start: float
    t_measure: Optional[float]=None
    n_repeat: Optional[int]=None
    interval: Optional[float]=None
    threshold : Optional[float]=None
    zero_on_high: bool=False


class acquisition_data(parent_data):
    def __init__(self):
        """
        init marker object
        Args:
            pulse_amplitude(double) : pulse amplitude in mV
        """
        super().__init__()
        self.data =  list()

        self.start_time = 0
        self.end_time = 0
        self._last_acquisition = -1

    def add_acquisition(self, acquisition):
        """
        add an acquisition
        Args:
            acquisition (acquisition): acquisition data object
        """
        acquisition.start += self.start_time
        if acquisition.start <= self._last_acquisition:
            raise Exception('Acquisition cannot be added at same time or before other acquisition')
        self._last_acquisition = acquisition.start
        self.data.append(acquisition)
        end_time = acquisition.start
        if end_time > self.end_time:
            self.end_time = end_time

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

    def append(self, other):
        '''
        Append two segments to each other, where the other segment is placed after the first segment.
        Args:
            other (acquisition_data) : other pulse data object to be appended

        '''
        end_time = self.end_time

        other_shifted = other._shift_all_time(end_time)
        self.data += other_shifted.data
        self.end_time += other.end_time

    def __copy__(self):
        """
        make a copy of this marker.
        """
        my_copy = acquisition_data()
        my_copy.data = copy.copy(self.data)
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
            raise ValueError("when shifting time, you cannot make negative times.")

        data_copy_shifted = copy.copy(self)

        for i in range(len(data_copy_shifted.data)):
            data_copy_shifted.my_data[i] = acquisition_data(data_copy_shifted.data[i].start + time_shift,
                                                            data_copy_shifted.data[i].t_mreasure)

        return data_copy_shifted

    def get_data(self):
        return self.data

    def print_all(self):
        for i in self.data:
            print(i)

    def get_vmin(self,sample_rate = 1e9):
        raise NotImplementedError()

    def get_vmax(self,sample_rate = 1e9):
        raise NotImplementedError()

    def integrate_waveform(self, sample_rate):
        raise NotImplementedError()

    def __add__(self, other):
        '''
        define addition operator for pulse_data object
        '''
        if not isinstance(other, acquisition_data):
            raise TypeError(f"Cannot add acquistion_data to type {type(other)}")
        new_data = self.__copy__()
        new_data.data += other.data
        return new_data

    def __mul__(self, rhs):
        raise NotImplementedError()

    def _render(self, sample_rate, ref_channel_states, LO):
        '''
        make a full rendering of the waveform at a predermined sample rate.
        '''
        # express in Gs/s
        sample_rate = sample_rate*1e-9
        sample_time_step = 1/sample_rate

        t_tot = self.total_time

        # get number of points that need to be rendered
        t_tot_pt = round_pt(t_tot, sample_time_step) + 1

        my_sequence = np.full([int(t_tot_pt)], None)

        for acq in self.data:
            start = round_pt(acq.start, sample_time_step)

            my_sequence[start] = 100

        return my_sequence[:-1]

    def get_metadata(self, name):
        metadata = {}
        acq_d = {}

        for i,acq in enumerate(self.data):
            acq_d[f'acq{i}'] = {
                'start':acq.start,
                't_measure':acq.t_measure,
                }

        if acq_d:
            metadata[name+'_acq'] = acq_d

        return metadata

def round_pt(t, t_sample):
    return int(t / t_sample + 0.5)

