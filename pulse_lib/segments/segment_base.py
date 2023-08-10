"""
File containing the parent class where all segment objects are derived from.
"""
import copy

import numpy as np
import matplotlib.pyplot as plt

from pulse_lib.segments.utility.data_handling_functions import loop_controller
from pulse_lib.segments.data_classes.data_generic import data_container
from pulse_lib.segments.utility.looping import loop_obj
from pulse_lib.segments.utility.setpoint_mgr import setpoint_mgr
from pulse_lib.segments.data_classes.data_generic import map_index
from pulse_lib.segments.utility.data_handling_functions import update_dimension


class segment_base():
    '''
    Class defining base function of a segment. All segment types should support all operators.
    If you make new data type, here you should buil-in in basic support to allow for general operations.

    For an example, look in the data classes files.
    '''
    def __init__(self, name, data_object, segment_type = 'render'):
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

        # store data in numpy looking object for easy operator access.
        self.data = data_container(data_object)

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
        self.is_slice = False

    def _copy(self, cpy):
        cpy.type = copy.copy(self.type)
        cpy.data = copy.copy(self.data)

        # note that the container objecet needs to take care of these. By default it will refer to the old references.
        cpy.reference_channels = copy.copy(self.reference_channels)
        cpy.IQ_ref_channels = copy.copy(self.IQ_ref_channels)
        cpy.references_markers = copy.copy(self.references_markers)

        # setpoints of the loops (with labels and units)
        cpy._setpoints = copy.copy(self._setpoints)

        return cpy

    @loop_controller
    def reset_time(self, time=None):
        '''
        resets the time back to zero after a certain point
        Args:
            time (double) : (optional), after time to reset back to 0. Note that this is absolute time and not rescaled time.
        '''
        self.data_tmp.reset_time(time)
        return self.data_tmp

    @loop_controller
    def wait(self, time, reset_time=False):
        '''
        resets the time back to zero after a certain point
        Args:
            time (double) : time in ns to wait
        '''
        if time < 0:
            raise Exception(f'Negative wait time {time} is not allowed')
        self.data_tmp.wait(time)
        if reset_time:
            self.data_tmp.reset_time(None)
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
            new_segment.data = new_segment.data + other.data

        elif type(other) == int or type(other) == float:
            new_segment.data += other
        else:
            raise TypeError("Please add up segment_single type or a number ")

        return new_segment

    def __iadd__(self, other):
        '''
        define addition operator for segment_single
        '''
        if isinstance(other, segment_base):
            self.data = self.data + other.data

        elif type(other) == int or type(other) == float:
            self.data += other
        else:
            raise TypeError("Please add up segment_single type or a number ")

        return self

    def __sub__(self, other):
        return self.__add__(other*-1)

    def __isub__(self, other):
        return self.__iadd__(other*-1)

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

        # To avoid unnecessary copying of data we first slice data, set self.data=None, copy, and then restore data in self.
        # This trick makes the indexing operation orders faster.
        data_org = self.data
        self.data = None
        item = copy.copy(self)
        self.data = data_org

        item.data = data_item
        item.is_slice = True
        return item

    def append(self, other):
        '''
        Append a segment to the end of this segment.
        '''
        self.add(other, time=-1)

    def add(self, other, time=None):
        '''
        Add the other segment behind this segment.
        Args:
            other (segment) : the segment to be appended
            time (double/loop_obj) : add at the given time. f None, append at total_time of the segment)

        A time reset will be done after the other segment is added.
        TODO: transfer of units
        '''
        if other.shape != (1,):
            other_loopobj = loop_obj()
            other_loopobj.add_data(other.data, axis=list(range(other.data.ndim -1,-1,-1)),
                                   dtype=object)
            self._setpoints += other._setpoints
            self.__add_segment(other_loopobj, time)
        else:
            self.__add_segment(other.data[0], time)

        return self

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
        if not isinstance(loop_obj, float):
            raise Exception(f'update_dim failed. Reload pulselib!')
        return self.data_tmp

    @loop_controller
    def __add_segment(self, other, time):
        """
        Add segment to this one. If time is not specified it will be added at start-time.

        Args:
            other (segment_base) : the segment to be appended
            time: time to add the segment.
        """
        self.data_tmp.add_data(other, time)
        return self.data_tmp

    # ==== getters on all_data

    @property
    def pulse_data_all(self):
        # TODO @@@: split virtual voltage gates from IQ channels. Combining only needed for virtual voltage.
        '''
        pulse data object that contains the counted op data of all the reference channels (e.g. IQ and virtual gates).
        '''
        if self._pulse_data_all is None:
            if (len(self.reference_channels) == 0
                and len(self.references_markers) == 0
                and len(self.IQ_ref_channels)== 0):
                self._pulse_data_all = self.data
            else:
                self._pulse_data_all = copy.copy(self.data)
                for ref_chan in self.reference_channels:
                    self._pulse_data_all = self._pulse_data_all + ref_chan.segment.pulse_data_all*ref_chan.multiplication_factor
                for ref_chan in self.IQ_ref_channels:
                    self._pulse_data_all = self.pulse_data_all + ref_chan.virtual_channel.get_IQ_data(ref_chan.out_channel)
                for ref_chan in self.references_markers:
                    self._pulse_data_all = self._pulse_data_all + ref_chan.IQ_channel_ptr.get_marker_data()

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
            ref_names = [ref.virtual_channel.name for ref in self.IQ_ref_channels]
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
            pulse_data_curr_seg = self.data[map_index(index, self.data.shape)]

        line = '-' if self.type == 'render' else ':'
        try:
            LO = self._qubit_channel.iq_channel.LO
        except:
            LO = None

        y = pulse_data_curr_seg.render(sample_rate, LO=LO)
        x = np.linspace(0, pulse_data_curr_seg.total_time, len(y))
        plt.plot(x, y, line, label=self.name)
        plt.xlabel("time (ns)")
        plt.ylabel("amplitude (mV)")
        plt.legend()

    def get_metadata(self):
        # Uses highest index of sequencer array (data_tmp)
        return self.data_tmp.get_metadata(self.name)
