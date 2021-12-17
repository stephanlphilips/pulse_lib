"""
File containing the parent class where all segment objects are derived from.
"""

import numpy as np
import matplotlib.pyplot as plt

from pulse_lib.segments.utility.data_handling_functions import loop_controller, get_union_of_shapes, update_dimension, find_common_dimension
from pulse_lib.segments.data_classes.data_generic import data_container
from pulse_lib.segments.utility.looping import loop_obj
from pulse_lib.segments.utility.setpoint_mgr import setpoint_mgr
from pulse_lib.segments.data_classes.data_generic import map_index

from functools import wraps
import copy



def last_edited(f):
    '''
    just a simpe decorater used to say that a certain wavefrom is updaded and therefore a new upload needs to be made to the awg.
    '''
    @wraps(f)
    def wrapper(*args, **kwargs):
        if args[0].render_mode == True:
            ValueError("cannot alter this segment, this segment ({}) in render mode.".format(args[0].name))
        args[0]._last_edit = last_edit.ToRender
        return f(*args, **kwargs)
    return wrapper

class last_edit:
    """
    spec of what the state is of the pulse.
    """
    ToRender = -1
    Rendered = 0

class segment_base():
    '''
    Class defining base function of a segment. All segment types should support all operators.
    If you make new data type, here you should buil-in in basic support to allow for general operations.

    For an example, look in the data classes files.
    '''
    def __init__(self, name, data_object, HVI_variable_data = None, segment_type = 'render'):
        '''
        Args:
            name (str): name of the segment usually the channel name
            data_object (object) : class that is used for saving the data type.
            HVI_variable_data (segment_HVI_variables) : segment used to keep variables that can be used in HVI.
            segment_type (str) : type of the segment (e.g. 'render' --> to be rendered, 'virtual'--> no not render.)
        '''
        self.type = segment_type
        self.name = name
        self.render_mode = False
        # variable specifing the laetest change to the waveforms,
        self._last_edit = last_edit.ToRender

        # store data in numpy looking object for easy operator access.
        self.data = data_container(data_object)
        self._data_hvi_variable = HVI_variable_data

        # references to other channels (for virtual gates).
        self.reference_channels = []
        # reference channels for IQ virtual channels
        self.IQ_ref_channels = []
        self.references_markers = []
        # local copy of self that will be used to count up the virtual gates.
        self._pulse_data_all = None
        # data caching variable. Used for looping and so on (with a decorator approach)
        self.data_tmp = None
        # variable specifing the lastest time the pulse_data_all is updated

        # setpoints of the loops (with labels and units)
        self._setpoints = setpoint_mgr()

    def _copy(self, cpy):
        cpy.type = copy.copy(self.type)
        cpy.data = copy.copy(self.data)
        # not full sure if this should be copied. Depends a bit on the usage scenario.
        cpy._data_hvi_variable = copy.copy(self._data_hvi_variable)

        # note that the container objecet needs to take care of these. By default it will refer to the old references.
        cpy.reference_channels = copy.copy(self.reference_channels)
        cpy.IQ_ref_channels = copy.copy(self.IQ_ref_channels)
        cpy.references_markers = copy.copy(self.references_markers)

        # setpoints of the loops (with labels and units)
        cpy._setpoints = copy.copy(self._setpoints)

        return cpy

    @last_edited
    @loop_controller
    def reset_time(self, time=None, extend_only = False):
        '''
        resets the time back to zero after a certain point
        Args:
            time (double) : (optional), after time to reset back to 0. Note that this is absolute time and not rescaled time.
            extend_only (bool) : will just extend the time in the segment and not reset it if set to true [do not use when composing wavoforms...].
        '''
        self.data_tmp.reset_time(time, extend_only)
        return self.data_tmp

    @last_edited
    @loop_controller
    def wait(self, time):
        '''
        resets the time back to zero after a certain point
        Args:
            time (double) : time in ns to wait
        '''
        self.data_tmp.wait(time)
        return self.data_tmp


    @property
    def setpoints(self):
        return self._setpoints

    def __add__(self, other):
        '''
        define addition operator for segment_single
        '''
        new_segment = copy.copy(self)
        if isinstance(other, segment_base):
            shape1 = new_segment.data.shape
            shape2 = other.data.shape
            new_shape = get_union_of_shapes(shape1, shape2)
            other_data = update_dimension(other.data, new_shape)
            new_segment.data= update_dimension(new_segment.data, new_shape)
            new_segment.data += other_data

        elif type(other) == int or type(other) == float:
            new_segment.data += other
        else:
            raise TypeError("Please add up segment_single type or a number ")

        return new_segment

    def __sub__(self, other):
        return self.__add__(other*-1)

    def __mul__(self, other):
        '''
        muliplication operator for segment_single
        '''
        new_segment = copy.copy(self)

        if isinstance(other, segment_base):
            raise TypeError("muliplication of two segments not supported. Please multiply by a number.")
        elif type(other) == int or type(other) == float or type(other) == np.double:
            new_segment.data *= other
        else:
            raise TypeError("Please add up segment_single type or a number ")

        return new_segment

    def __truediv__(self, other):
        raise NotImplemented

    def __getitem__(self, *key):
        '''
        get slice or single item of this segment (note no copying, just referencing)
        Args:
            *key (int/slice object) : key of the element -- just use numpy style accessing (slicing supported)
        '''
        data_item = self.data[key[0]]
        if not isinstance(data_item, data_container):
            # If the slice contains only 1 element, then it's not a data_container anymore.
            # Put it in a data_container to maintain pulse_lib structure.
            data_item = data_container(data_item)

        # To avoid unnecessary copying of data we first slice on self, copy, and then restore data in self.
        # This trick makes the indexing operation orders faster.
        data_org = self.data
        self.data = data_item
        item = copy.copy(self)
        item.data = data_item # TODO [SdS]: make clean solution
        self.data = data_org
        return item

    @last_edited
    def append(self, other, time = None):
        '''
        Put the other segment behind this one.
        Args:
            other (segment_single) : the segment to be appended
            time (double/loop_obj) : attach at the given time (if None, append at total_time of the segment)

        A time reset will be done after the other segment is added.
        TODO: transfer of units
        '''
        other_loopobj = loop_obj()
        other_loopobj.add_data(other.data, axis=list(range(other.data.ndim -1,-1,-1)))
        self._setpoints += other._setpoints
        self.__append(other_loopobj, time)

        return self

    @last_edited
    @loop_controller
    def repeat(self, number):
        '''
        repeat a waveform n times.

        Args:
            number (int) : number of ties to repeat the waveform
        '''

        data_copy = copy.copy(self.data_tmp)
        for i in range(number-1):
            self.data_tmp.append(data_copy)

        return self.data_tmp

    @loop_controller
    def update_dim(self, loop_obj):
        '''
        update the dimesion of the segment by providing a loop object to it (decorator takes care of it).

        Args:
            loop_obj (loop_obj) : loop object with certain dimension to add.
        '''
        return self.data_tmp

    def add_HVI_marker(self, marker_name, t_off = 0):
        '''
        Add a HVI marker that corresponds to the current time of the segment (defined by reset_time).

        Args:
            marker_name (str) : name of the marker to add
            t_off (str) : offset to be given from the marker
        '''
        times = loop_obj(no_setpoints=True)
        times.add_data(self.data.start_time, axis=list(range(self.data.ndim -1,-1,-1)))

        self.add_HVI_variable(marker_name, times + t_off, True)

    def add_HVI_variable(self, marker_name, value, time=False):
        """
        Add time for the marker.

        Args:
            name (str) : name of the variable

            value (double) : value to assign to the variable

            time (bool) : if the value is a timestamp (determines behaviour when the variable is used in a sequence) (coresponding to a master clock)
        """
        self._data_hvi_variable._add_HVI_variable(marker_name, value, time)

    @loop_controller
    def __append(self, other, time):
        """
        Put the other segment behind this one (for single segment data object)

        Args:
            other (segment_single) : the segment to be appended
            time (double/loop_obj) : attach at the given time (if None, append at total_time of the segment)
        """
        if time is None:
            time = self.data_tmp.total_time


        self.data_tmp.append(other, time)
        return self.data_tmp

    @last_edited
    @loop_controller
    def slice_time(self, start_time, stop_time):
        """
        Cuts parts out of a segment.

        Args:
            start_time (double) : effective new start time
            stop_time (double) : new ending time of the segment

        The slice_time function allows you to cut a waveform in different sizes.
        This function should be handy for debugging, example usage would be,
        You are runnning an algorithm and want to check what the measurement outcomes are though the whole algorithm.
        Pratically, you want to know
            0 -> 10ns (@10 ns still everything as expected?)
            0 -> 20ns
            0 -> ...
        This function would allow you to do that, e.g. by calling segment.cut_segment(0, lp.linspace(10,100,9))
        """
        self.data_tmp.slice_time(start_time, stop_time)
        return self.data_tmp

    # ==== getters on all_data

    @property
    def pulse_data_all(self):
        # TODO @@@: split virtual voltage gates from IQ channels. Combining only needed for virtual voltage.
        '''
        pulse data object that contains the counted op data of all the reference channels (e.g. IQ and virtual gates).
        '''
        if self.last_edit == last_edit.ToRender or self._pulse_data_all is None:
            self._pulse_data_all = copy.copy(self.data)
            for ref_chan in self.reference_channels:
                # make sure both have the same size.
                my_shape = find_common_dimension(self._pulse_data_all.shape, ref_chan.segment.shape)
                self._pulse_data_all = update_dimension(self._pulse_data_all, my_shape)
                self._pulse_data_all += ref_chan.segment.data*ref_chan.multiplication_factor
                ref_chan.segment._last_edit = last_edit.Rendered
            for ref_chan in self.IQ_ref_channels:
                # todo -- update dim functions
                my_shape = find_common_dimension(self._pulse_data_all.shape, ref_chan.virtual_channel_pointer.shape)
                self._pulse_data_all = update_dimension(self._pulse_data_all, my_shape)
                self._pulse_data_all += ref_chan.virtual_channel_pointer.get_IQ_data(ref_chan.LO, ref_chan.IQ_render_option, ref_chan.image_render_option)
            for ref_chan in self.references_markers:
                my_shape = find_common_dimension(self._pulse_data_all.shape, ref_chan.IQ_channel_ptr.shape)
                self._pulse_data_all = update_dimension(self._pulse_data_all, my_shape)
                self._pulse_data_all += ref_chan.IQ_channel_ptr.get_marker_data()

            self._last_edit = last_edit.Rendered

        return self._pulse_data_all

    @property
    def shape(self):
        if self.render_mode == False:
            return self.data.shape
        else:
            return self.pulse_data_all.shape

    @property
    def ndim(self):
        if self.render_mode == False:
            return self.data.ndim
        else:
            return self.pulse_data_all.shape

    @property
    def total_time(self):
        if self.render_mode == False:
            return self.data.total_time
        else:
            return self.pulse_data_all.total_time

    def get_total_time(self, index):
        return self.total_time[map_index(index, self.shape)]

    @property
    def start_time(self):
        if self.render_mode == False:
            return self.data.start_time
        else:
            return self.pulse_data_all.start_time

    # ==== operations working on an index

    def _get_data_all_at(self, index):
        return self.pulse_data_all[map_index(index, self.shape)]

    def get_segment(self, index, sample_rate=1e9, ref_channel_states=None):
        '''
        get the numpy output of as segment

        Args:
            index of segment (list) : which segment to render (e.g. [0] if dimension is 1 or [2,5,10] if dimension is 3)
            sample_rate (float) : #/s (number of samples per second)

        Returns:
            A numpy array that contains the points for each ns
            points is the expected lenght.
        '''
        if ref_channel_states:
            # Filter reference channels for use in data_pulse cache
            ref_channel_states = copy.copy(ref_channel_states)
            ref_channel_states.start_phases_all = None
            ref_names = [ref.virtual_channel_name for ref in self.IQ_ref_channels]
            ref_channel_states.start_phase = {key:value
                                              for (key,value) in ref_channel_states.start_phase.items()
                                              if key in ref_names}

        return self._get_data_all_at(index).render(sample_rate, ref_channel_states)

    def v_max(self, index, sample_rate = 1e9):
        return self._get_data_all_at(index).get_vmax(sample_rate)

    def v_min(self, index, sample_rate = 1e9):
        return self._get_data_all_at(index).get_vmin(sample_rate)

    def integrate(self, index, sample_rate = 1e9):
        '''
        Get integral value of the waveform (e.g. to calculate an automatic compensation)

        Args:
            index (tuple) : index of the concerning waveform
            sample_rate (double) : rate at which to render the pulse

        Returns:
            integral (float) : integral of the pulse
        '''
        return self._get_data_all_at(index).integrate_waveform(sample_rate)

    def plot_segment(self, index = [0], render_full = True, sample_rate = 1e9):
        '''
        Args:
            index : index of which segment to plot
            render full (bool) : do full render (e.g. also get data form virtual channels). Put True if you want to see the waveshape send to the AWG.
            sample_rate (float): standard 1 Gs/s
        '''
        if render_full == True:
            pulse_data_curr_seg = self._get_data_all_at(index)
        else:
            pulse_data_curr_seg = self.data[map_index(index, self.shape)]

        line = '-' if self.type == 'render' else ':'
        y = pulse_data_curr_seg.render(sample_rate)
        x = np.linspace(0, pulse_data_curr_seg.total_time, len(y))
        plt.plot(x, y, line, label=self.name)
        plt.xlabel("time (ns)")
        plt.ylabel("amplitude (mV)")
        plt.legend()

    @property
    def last_edit(self):
        for i in self.reference_channels:
            if i.segment._last_edit == last_edit.ToRender:
                self._last_edit = last_edit.ToRender
        for i in self.IQ_ref_channels:
            if i.virtual_channel_pointer  == last_edit.ToRender:
                self._last_edit = last_edit.ToRender
        for i in self.references_markers:
            if i.IQ_channel_ptr  == last_edit.ToRender:
                self._last_edit = last_edit.ToRender

        return self._last_edit

    def get_metadata(self):
        # Uses highest index of sequencer array (data_tmp)
        return self.data_tmp.get_metadata(self.name)
