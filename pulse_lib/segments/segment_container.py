"""
File contains an object that mananges segments. E.g. you are dealing with mutiple channels.
This object also allows you to do operations on all segments at the same time.
"""

from pulse_lib.segments.segment_pulse import segment_pulse
from pulse_lib.segments.segment_IQ import segment_IQ
from pulse_lib.segments.segment_markers import segment_marker
from pulse_lib.segments.segment_HVI_variables import segment_HVI_variables
from pulse_lib.segments.segment_acquisition import segment_acquisition
from pulse_lib.segments.segment_measurements import segment_measurements

import pulse_lib.segments.utility.looping as lp
from pulse_lib.segments.utility.data_handling_functions import find_common_dimension, update_dimension, reduce_arr, upconvert_dimension
from pulse_lib.segments.utility.setpoint_mgr import setpoint_mgr
import uuid

import numpy as np
import datetime
import copy
from dataclasses import dataclass


class segment_container():
    '''
    Class containing all the single segments for for a series of channels.
    This is a organisational class.
    Class is capable of termining what volatages are required for each channel.
    Class returns vmin/vmax data to awg object
    Class returns upload data as a numpy<double> array to awg object.
    '''
    def __init__(self, channel_names, markers=[], virtual_gates_objs=[], IQ_channels_objs=[],
                 digitizer_channels = [],
                 name=None, sample_rate=None):
        """
        initialize a container for segments.
        Args:
            channel_names (list<str>) : list with names of physical output channels on the AWG
            markers (list<str>) : declaration which of these channels are markers
            virtual_gates_objs (list<virtual_gates_constructor>) : list of object that define virtual gates
            IQ_channels_objs (list<IQ_channel_constructor>) : list of objects that define virtual IQ channels.
            Name (str): Optional name of segment container
            sample_rate (float): Optional sample rate of segment container. This sample rate overrules the default set on the sequence.
        """
        # physical + virtual channels + digitizer channels
        self.channels = []
        self.render_mode = False
        self._total_times = None
        self.id = uuid.uuid4()
        self._Vmin_max_data = dict()
        self._software_markers = segment_HVI_variables("HVI_markers")
        self._segment_measurements = segment_measurements()

        self._virtual_gates_objs = virtual_gates_objs
        self._IQ_channel_objs = IQ_channels_objs
        self._digitizer_channels = digitizer_channels
        self.name = name
        self.sample_rate = sample_rate

        for name in channel_names:
            self._Vmin_max_data[name] = {"v_min" : None, "v_max" : None}

        self.prev_upload = datetime.datetime.utcfromtimestamp(0)


        # define real channels (+ markers)
        for name in channel_names:
            setattr(self, name, segment_pulse(name, self._software_markers))
            self.channels.append(name)
        for name in markers:
            setattr(self, name, segment_marker(name, self._software_markers))
            self.channels.append(name)

        # define virtual gates
        for virtual_gates in virtual_gates_objs:
            # make segments for virtual gates.
            for virtual_gate_name in virtual_gates.virtual_gate_names:
                setattr(self, virtual_gate_name, segment_pulse(virtual_gate_name, self._software_markers, 'virtual_baseband'))
                self.channels.append(virtual_gate_name)

        # define virtual IQ channels
        for IQ_channels_obj in IQ_channels_objs:
            for qubit_channel in IQ_channels_obj.qubit_channels:
                setattr(self, qubit_channel.channel_name, segment_IQ(qubit_channel.channel_name, self._software_markers))
                self.channels.append(qubit_channel.channel_name)

        # add the reference between channels for baseband pulses (->virtual gates) and IQ channels.
        add_reference_channels(self, self._virtual_gates_objs, self._IQ_channel_objs)

        for digitizer_channel in self._digitizer_channels:
            name = digitizer_channel.name
            setattr(self, name, segment_acquisition(name, self._segment_measurements))
            self.channels.append(name)

        self._setpoints = setpoint_mgr()

    def __getitem__(self, index):
        if isinstance(index, str):
            if index not in self.channels:
                raise ValueError(f'Unknown channel {index}')
            return getattr(self, index)
        elif isinstance(index, int):
            new = segment_container([])

            new._virtual_gates_objs = self._virtual_gates_objs
            new._IQ_channel_objs = self._IQ_channel_objs

            new.channels = self.channels

            self.extend_dim(self.shape)

            for chan_name in self.channels:
                chan = getattr(self, chan_name)
                new_chan = chan[index]
                setattr(new, chan_name,new_chan)

            new.render_mode = copy.copy(self.render_mode)
            new._Vmin_max_data = copy.copy(self._Vmin_max_data)
            new._software_markers =self._software_markers
            new._setpoints = self._setpoints

            # update the references in of all the channels
            add_reference_channels(new, self._virtual_gates_objs, self._IQ_channel_objs)

            return new



    def __copy__(self):
        new = segment_container([])

        new._virtual_gates_objs = self._virtual_gates_objs
        new._IQ_channel_objs = self._IQ_channel_objs

        new.channels = copy.copy(self.channels)

        for chan_name in self.channels:
            chan = getattr(self, chan_name)
            new_chan = copy.copy(chan)
            setattr(new, chan_name,new_chan)

        new.render_mode = copy.copy(self.render_mode)
        new._Vmin_max_data = copy.copy(self._Vmin_max_data)
        new._software_markers = copy.copy(self._software_markers)
        new._setpoints = copy.copy(self._setpoints)

        # update the references in of all the channels
        add_reference_channels(new, self._virtual_gates_objs, self._IQ_channel_objs)

        return new

    @property
    def software_markers(self):
        return self._software_markers

    @software_markers.setter
    def software_markers(self, new_marker):
        self._software_markers = new_marker
        add_reference_channels(self, self._virtual_gates_objs, self._IQ_channel_objs)

    @property
    def measurements(self):
        return self._segment_measurements.measurements

    @property
    def acquisitions(self):
        result = {}
        for digitizer_channel in self._digitizer_channels:
            name = digitizer_channel.name
            result[name] = self[name].pulse_data_all
        return result

    @property
    def shape(self):
        '''
        get combined shape of all the waveforms
        '''
        my_shape = (1,)
        for i in self.channels:
            dim = getattr(self, i).shape
            my_shape = find_common_dimension(my_shape, dim)

        return my_shape

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def last_mod(self):
        time = datetime.datetime.utcfromtimestamp(0)
        for i in self.channels:
            if getattr(self, i).last_edit > time:
                time = getattr(self, i).last_edit
        return time

    @property
    def total_time(self):
        '''
        get the total time that will be uploaded for this segment to the AWG
        Returns:
            times (np.ndarray) : numpy array with the total time (maximum of all the channels), for all the different loops executed.
        '''
        if self.render_mode and self._total_times is not None:
            return self._total_times

        shape = list(self.shape)
        n_channels = len(self.channels)

        time_data = np.empty([n_channels] + shape)

        for i, channel in enumerate(self.channels):
            time_data[i] = upconvert_dimension(self[channel].total_time, shape)

        times = np.amax(time_data, axis = 0)

        if self.render_mode:
            self._total_times = times

        return times

    @property
    def _start_time(self):
        '''
        get the total time that will be uploaded for this segment to the AWG
        Returns:
            times (np.ndarray) : numpy array with the total time (maximum of all the channels), for all the different loops executed.
        '''

        shape = list(self.shape)
        n_channels = len(self.channels)

        time_data = np.empty([n_channels] + shape)

        for i in range(len(self.channels)):
            time_data[i] = upconvert_dimension(getattr(self, self.channels[i]).start_time, shape)

        times = np.amax(time_data, axis = 0)

        return times

    @property
    def setpoint_data(self):

        comb_setpoints = copy.deepcopy(self._setpoints)

        for channel in self.channels:
            segment = self[channel]
            comb_setpoints += segment.setpoints

        return comb_setpoints


    @property
    def Vmin_max_data(self):
        if self.prev_upload < self.last_mod:

            for i in range(len(self.channels)):
                self._Vmin_max_data[self.channels[i]]['v_min'] = getattr(self,self.channels[i]).v_min
                self._Vmin_max_data[self.channels[i]]['v_max'] = getattr(self,self.channels[i]).v_max

        return self._Vmin_max_data

    def append(self, other, time=None):
        '''
        append other segments the the current ones in the container.
        Args:
            other (segment_container) : other segment to append
        '''
        if not isinstance(other, segment_container):
            raise TypeError("segment_container object expected. Did you supply a single segment?")

        # make sure all object have a full size.
        my_shape = find_common_dimension(self.shape, other.shape)
        other.extend_dim(my_shape)
        self.extend_dim(my_shape)

        self._setpoints += other._setpoints

        if time == None:
            times = self.total_time

            time = lp.loop_obj(no_setpoints=True)
            time.add_data(times, list(range(len(times.shape)-1, -1,-1)))
        for i in self.channels:
            segment = getattr(self, i)
            segment.append(getattr(other, i), time)

    def slice_time(self, start, stop):
        """
        slice time in a segment container
        Args:
            start (double) : start time of the slice
            stop (double) : stop time of the slice

        The slice_time function allows you to cut all the waveforms in the segment container in different sizes.
        This function should be handy for debugging, example usage would be,
        You are runnning an algorithm and want to check what the measurement outcomes are though the whole algorithm.
        Pratically, you want to know
            0 -> 10ns (@10 ns still everything as expected?)
            0 -> 20ns
            0 -> ...
        This function would allow you to do that, e.g. by calling segment_container.cut_segment(0, lp.linspace(10,100,9))
        """
        for i in self.channels:
            segment = getattr(self, i)
            segment.slice_time(start, stop)

    def reset_time(self, extend_only = False):
        '''
        Args:
            extend_only (bool) : will just extend the time in the segment and not reset it if set to true [do not use when composing wavoforms...].

        Allings all segments togeter and sets the input time to 0,
        e.g. ,
        chan1 : waveform until 70 ns
        chan2 : waveform until 140ns
        -> totaltime will be 140 ns,
        when you now as a new pulse (e.g. at time 0, it will actually occur at 140 ns in both blocks)
        '''

        n_channels = len(self.channels)
        shape = list(self.shape)
        time_data = np.empty([n_channels] + shape)

        for i in range(len(self.channels)):
            time_data[i] = upconvert_dimension(getattr(self, self.channels[i]).total_time, shape)

        times = np.amax(time_data, axis = 0)
        times, axis = reduce_arr(times)
        if len(axis) == 0:
            loop_obj = times
        else:
            loop_obj = lp.loop_obj(no_setpoints=True)
            loop_obj.add_data(times, axis)

        for i in self.channels:
            segment = getattr(self, i)
            segment.reset_time(loop_obj, False)

    def get_waveform(self, channel, index = [0], sample_rate=1e9, ref_channel_states=None):
        '''
        function to get the raw data of a waveform,
        inputs:
            channel (str) : channel name of the waveform you want
            index (tuple) :
        returns:
            np.ndarray[ndim=1, dtype=double] : waveform as a numpy array
        '''
        return getattr(self, channel).get_segment(index, sample_rate, ref_channel_states)

    def extend_dim(self, shape=None, ref = False):
        '''
        extend the dimensions of the waveform to a given shape.
        Args:
            shape (tuple) : shape of the new waveform
            ref (bool) : put to True if you want to extend the dimension by using pointers instead of making full copies.
        If referencing is True, a pre-render will already be performed to make sure nothing is rendered double.
        '''
        if shape is None:
            shape = self.shape


        for i in self.channels:
            if self.render_mode == False:
                getattr(self, i).data = update_dimension(getattr(self, i).data, shape, ref)

            if getattr(self, i).type == 'render' and self.render_mode == True:
                getattr(self, i)._pulse_data_all = update_dimension(getattr(self, i)._pulse_data_all, shape, ref)

        if self.render_mode == False:
            self._software_markers.data = update_dimension(self._software_markers.data, shape, ref)
        else:
            self._software_markers._pulse_data_all = update_dimension(self._software_markers.pulse_data_all, shape, ref)

    def add_block(self, start, stop, channels, amplitudes):
        '''
        Adds a block to each of the specified channels.
        Args:
           start (float, loop_obj): start of the block
           stop (float, loop_obj): stop of the block
           channels (List[str]): channels to apply the block to
           amplitudes (List[float, loop_obj]): amplitude per channel
        '''
        for channel, amplitude in zip(channels, amplitudes):
            self[channel].add_block(start, stop, amplitude)

    def add_ramp(self, start, stop, channels, start_amplitudes, stop_amplitudes):
        '''
        Adds a ramp to each of the specified channels.
        Args:
           start (float, loop_obj): start of the block
           stop (float, loop_obj): stop of the block
           channels (List[str]): channels to apply the block to
           start_amplitudes (List[float, loop_obj]): start amplitude per channel
           stop_amplitudes (List[float, loop_obj]): stop amplitude per channel
        '''
        for channel, start_amp, stop_amp in zip(channels, start_amplitudes, stop_amplitudes):
            self[channel].add_ramp_ss(start, stop, start_amp, stop_amp)

    def add_HVI_marker(self, marker_name, t_off = 0):
        '''
        add a HVI marker that corresponds to the current time of the segment (defined by reset_time).

        Args:
            marker_name (str) : name of the marker to add
            t_off (str) : offset to be given from the marker
        '''
        times = lp.loop_obj(no_setpoints=True)
        # Look into this inversion of the setpoints
        times.add_data(self._start_time, axis=list(range(self.ndim -1,-1,-1)))

        self.add_HVI_variable(marker_name, times + t_off, True)

    def add_HVI_variable(self, marker_name, value, time=False):
        """
        add time for the marker.
        Args:
            name (str) : name of the variable
            value (double) : value to assign to the variable
            time (bool) : if the value is a timestamp (determines behaviour when the variable is used in a sequence) (coresponding to a master clock)
        """
        self._software_markers._add_HVI_variable(marker_name, value, time)

    def add_measurement_expression(self, expression=None, name=None, accept_if=None):
        '''
        Adds a measurement expression
        Args:
            TODO
        '''
        self._segment_measurements.add_expression(expression, accept_if=accept_if, name=name)

    def enter_rendering_mode(self):
        '''
        put the segments into rendering mode, which means that they cannot be changed. All segments will get their final length at this moment.
        '''
        self.reset_time()
        self.render_mode = True
        for i in self.channels:
            getattr(self, i).render_mode =  True
            # make a pre-render all all the pulse data (e.g. compose channels, do not render in full).
            if getattr(self, i).type == 'render':
                getattr(self, i).pulse_data_all

    def add_master_clock(self, time):
        '''
        add a master clock to the segment.

        Args:
            time (float) :  effective time that this segment start in the sequence.

        At the moment only correction for the HVI marker, clock for the MW signal needs to be implemented here later.
        '''

        self._software_markers._add_global_time_shift(time)

    def exit_rendering_mode(self):
        '''
        exit rendering mode and clear all the ram that was used for the rendering.
        '''
        self.render_mode = False
        self._total_times = None
        for i in self.channels:
            getattr(self, i).render_mode =  False
            getattr(self, i)._pulse_data_all = None

    def plot(self, index=(0,), channels=None, sample_rate=1e9):
        '''
        Plots selected channels.
        Args:
            index (tuple): loop index
            channels (list[str]): channels to plot, if None plot all channels.
            sample_rate (float): sample rate to use in rendering.
        '''
        if channels is None:
            channels = self.channels
        self.enter_rendering_mode()
        for channel in channels:
            self[channel].plot_segment(index, sample_rate=sample_rate)
        self.exit_rendering_mode()

    def get_metadata(self):
        '''
        get_metadata
        '''
        metadata = {}
        metadata['_total_time'] = self.total_time
        for ch in self.channels:
            metadata.update(self[ch].get_metadata())
        return metadata

@dataclass
class virtual_pulse_channel_info:
    """
    info that is needed to link a real channel to a virtual channel
    """
    name: str
    multiplication_factor: float
    seg_container: any

    @property
    def segment(self):
        return getattr(self.seg_container, self.name)


def add_reference_channels(segment_container_obj, virtual_gates_objs, IQ_channels_objs):
    '''
    add/update the references to the channels

    Args:
        segment_container_obj (segment_container) :
        virtual_gates_objs (list<virtual_gates_constructor>) : list of object that define virtual gates
        IQ_channels_objs (list<IQ_channel_constructor>) : list of objects that define virtual IQ channels.
    '''
    for channel in segment_container_obj.channels:
        getattr(segment_container_obj, channel).reference_channels = list()
        getattr(segment_container_obj, channel).IQ_ref_channels = list()
        getattr(segment_container_obj, channel).add_reference_markers = list()
        getattr(segment_container_obj, channel)._data_hvi_variable = segment_container_obj._software_markers

    for virtual_gates in virtual_gates_objs:
        # add reference in real gates.
        for i in range(virtual_gates.size):
            real_channel = getattr(segment_container_obj, virtual_gates.real_gate_names[i])
            virtual_gates_values = virtual_gates.virtual_gate_matrix_inv[i,:]

            for j in range(virtual_gates.size):
                multiplier = virtual_gates_values[j]
                # only add reference when it has a significant contribution
                if abs(multiplier) > 2E-4:
                    virtual_channel_reference_info = virtual_pulse_channel_info(virtual_gates.virtual_gate_names[j],
                        multiplier, segment_container_obj)
                    real_channel.add_reference_channel(virtual_channel_reference_info)

    # define virtual IQ channels
    for IQ_channels_obj in IQ_channels_objs:
        # set up maping to real IQ channels:
        for IQ_out_channel in IQ_channels_obj.IQ_out_channels:
            real_channel = getattr(segment_container_obj, IQ_out_channel.awg_channel_name)
            for qubit_channel in IQ_channels_obj.qubit_channels:
                virtual_channel = getattr(segment_container_obj, qubit_channel.channel_name)
                real_channel.add_IQ_channel(IQ_channels_obj.LO, qubit_channel.channel_name,
                                            virtual_channel, IQ_out_channel.IQ_comp, IQ_out_channel.image)

        # set up markers
        for marker_name in IQ_channels_obj.marker_channels:
            real_channel_marker = getattr(segment_container_obj, marker_name)

            for qubit_channel in IQ_channels_obj.qubit_channels:
                virtual_channel = getattr(segment_container_obj, qubit_channel.channel_name)
                real_channel_marker.add_reference_marker_IQ(virtual_channel)


if __name__ == '__main__':
    import pulse_lib.segments.utility.looping as lp
    import matplotlib.pyplot as plt

    seg = segment_container(["a", "b", "c", "d"])
    # b = segment_container(["a", "b"])
    chan = segment_IQ("q1")
    setattr(seg, "q1", chan)

    chan_M = segment_marker("M1")
    setattr(seg, "M1", chan_M)
    seg.channels.append("q1")
    seg.channels.append("M1")
    # print(seg.channels)
    # print(seg.q1)
    seg.a.add_IQ_channel(1e9, "q1", chan, "I", "0")
    seg.b.add_IQ_channel(1e9, "q1", chan, "Q", "0")

    seg.M1.add_reference_marker_IQ(chan)

    seg['c'].add_block(0, 10, 100)
    seg.add_block(10, 20, ['c', 'd'], [-100, -50])
    seg.add_ramp(10, 20, ['d', 'c'], [50, -100], [100, 250])
    seg.add_ramp(0, 20, ['c', 'd'], [250, 100], [0,0])


    seg.a.add_block(0,lp.linspace(50,100,10),100)
    seg.a += 500
    seg.b += 500
    seg.reset_time()
    seg.q1.add_MW_pulse(0,100,10,1.015e9)
    seg.q1.wait(10)
    seg.reset_time()
    seg.q1.add_chirp(0,100,1e7,1.1e8, 100)
    seg.q1.wait(20)
    seg.q1.reset_time()
    seg.q1.add_chirp(0,100,1.1e9,1.e9, 100)
    seg.q1.wait(10)
    seg.add_HVI_marker("my_test")
    # print(seg._software_markers.data)
    # print(seg.setpoint_data)
    # print(a.a.data[2,2,2])

    seg.q1.plot_segment([0], sample_rate = 1e10)
    seg.a.plot_segment([0], True, sample_rate = 1e10)
    seg.b.plot_segment([0], True, sample_rate = 1e10)
    seg.c.plot_segment([0], True, sample_rate = 1e10)
    seg.d.plot_segment([0], True, sample_rate = 1e10)
    seg.M1.plot_segment([0])
    plt.show()

    plt.figure()
    seg.plot([0], ['c','d'])