"""
File contains an object that mananges segments. E.g. you are dealing with mutiple channels.
This object also allows you to do operations on all segments at the same time.
"""

from pulse_lib.segments.segment_HVI_variables import segment_HVI_variables
from pulse_lib.segments.segment_measurements import segment_measurements
from pulse_lib.segments.segment_container import segment_container
from pulse_lib.segments.data_classes.data_generic import map_index

import pulse_lib.segments.utility.looping as lp
from pulse_lib.segments.utility.data_handling_functions import find_common_dimension, reduce_arr
from pulse_lib.segments.utility.setpoint_mgr import setpoint_mgr

import numpy as np
import logging
import copy
from typing import List

logger = logging.getLogger(__name__)

class conditional_segment:

    def __init__(self, condition, branches:List[segment_container],
                 name=None, sample_rate=None):
        # create N segment_container
        self.condition = condition
        self.branches = branches

        # software markers and measurements stay empty
        self._software_markers = segment_HVI_variables("HVI_markers")
        self._segment_measurements = segment_measurements()

        # sample_rate must be equal

        # set sample rate (for all)

        self.name = name
        self.sample_rate = sample_rate

        # setpoints?
        self._setpoints = setpoint_mgr()

    def __copy__(self):
        pass

#    @property
#    def channels(self):
#        return self.branches[0].channels

    @property
    def software_markers(self):
        return self._software_markers

    @property
    def measurements(self):
        return self.branches[0].measurements

    @property
    def acquisitions(self):
        return self.branches[0].acquisitions

    @property
    def shape(self):
        '''
        get combined shape of all the waveforms
        '''
        shape = (1,)

        for branch in self.branches:
            shape = find_common_dimension(shape, branch.shape)

        return shape

#    @property
#    def ndim(self):
#        pass
#
    @property
    def total_time(self):
        shape = self.shape

        n_branches = len(self.branches)
        time_data = np.empty((n_branches,) + shape)

        for i, branch in enumerate(self.branches):
            time_data[i] = branch.total_time

        times = np.amax(time_data, axis = 0)

        return times

    def get_total_time(self, index):
        return self.total_time[map_index(index, self.shape)]

#    @property
#    def _start_time(self):
#        pass
#
    @property
    def setpoint_data(self):
        comb_setpoints = copy.deepcopy(self._setpoints)

        for branch in self.branches:
            comb_setpoints += branch.setpoint_data

        return comb_setpoints

    def reset_time(self):
        '''
        Alligns all segments together and sets the input time to 0,
        e.g. ,
        chan1 : waveform until 70 ns
        chan2 : waveform until 140ns
        -> totaltime will be 140 ns,
        when you now as a new pulse (e.g. at time 0, it will actually occur at 140 ns in both blocks)
        '''
        n_branches = len(self.branches)
        n_channels = len(self.branches[0].channels)
        shape = self.shape
        time_data = np.empty((n_branches*n_channels,) + shape)

        for ibranch, branch in enumerate(self.branches):
            for ich, ch in enumerate(branch.channels):
                time_data[ibranch * n_channels + ich] = branch[ch].total_time

        times = np.amax(time_data, axis=0)
        times, axis = reduce_arr(times)
        logger.info(f'times {times}')
        if len(axis) == 0:
            loop_obj = times
        else:
            loop_obj = lp.loop_obj(no_setpoints=True)
            loop_obj.add_data(times, axis)

        for branch in self.branches:
            for ch in branch.channels:
                branch[ch].reset_time(loop_obj)

    def enter_rendering_mode(self):
        self.reset_time()
        for branch in self.branches:
            branch.enter_rendering_mode()

    def add_master_clock(self, time):
        for branch in self.branches:
            branch.add_master_clock(time)

    def exit_rendering_mode(self):
        for branch in self.branches:
            branch.exit_rendering_mode()

    def plot(self, index=(0,), channels=None, sample_rate=1e9, render_full=False):
        pass

    def get_metadata(self):
        pass

