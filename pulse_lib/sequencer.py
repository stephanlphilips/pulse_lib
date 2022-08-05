import matplotlib.pyplot as pt
from qcodes import Parameter

from .schedule.hardware_schedule import HardwareSchedule
from .segments.conditional_segment import conditional_segment
from .segments.data_classes.data_HVI_variables import marker_HVI_variable
from .segments.data_classes.data_generic import data_container, parent_data
from .segments.segment_container import segment_container
from .segments.utility.data_handling_functions import find_common_dimension, update_dimension
from .segments.utility.setpoint_mgr import setpoint_mgr
from .segments.utility.looping import loop_obj
from .segments.utility.measurement_ref import MeasurementRef
from .measurements_description import measurements_description
from .acquisition.acquisition_conf import AcquisitionConf
from .acquisition.acquisition_param import AcquisitionParam
from .acquisition.player import SequencePlayer
from .acquisition.measurement_converter import MeasurementConverter, DataSelection, MeasurementParameter

from si_prefix import si_format

from typing import List
from collections.abc import Iterable
from numbers import Number
import numpy as np
import uuid
import logging

class sequencer():
    """
    Class to make sequences for segments.
    """
    def __init__(self, upload_module, digitizer_channels):
        '''
        make a new sequence object.
        Args:
            upload_module (uploader) : class of the upload module. Used to submit jobs
        Returns:
            None
        '''
        # each segment had its own unique identifier.
        self.id = uuid.uuid4()

        self._units = None
        self._setpoints = None
        self._names = None

        self._shape = (1,)
        self._sweep_index = [0]
        self.sequence = list()
        self.uploader = upload_module
        self._digitizer_channels = digitizer_channels

        self._measurements_description = measurements_description(digitizer_channels)

        # arguments of post processing the might be needed during rendering.
        self.neutralize = True

        # hardware schedule (HVI)
        self.hw_schedule = None

        self._n_rep = 1000
        self._sample_rate = 1e9
        self._HVI_variables = None
        self._alignment = None
        self._acquisition_conf = AcquisitionConf()
        self._measurement_converter = None

    @property
    def n_rep(self):
        return self._n_rep

    @n_rep.setter
    def n_rep(self, value):
        if self._measurement_converter is not None:
            raise Exception('Cannot change n_rep after calling get_measurement_results or '
                            'get_measurement_param')
        self._n_rep = value

    @property
    def sweep_index(self):
        return self._sweep_index

    @property
    def shape(self):
        return self._shape

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def setpoint_data(self):
        setpoint_data = setpoint_mgr()
        for seg_container in self.sequence:
            setpoint_data += seg_container.setpoint_data

        return setpoint_data

    @property
    def units(self):
        return self.setpoint_data.units

    @property
    def labels(self):
        return self.setpoint_data.labels

    @property
    def setpoints(self):
        return self.setpoint_data.setpoints

    @property
    def HVI_variables(self):
        """
        object that contains variable that can be ported into HVI.
        """
        return self._HVI_variables

    @property
    def sample_rate(self):
        return self._sample_rate

    @sample_rate.setter
    def sample_rate(self, rate):
        """
        Rate at which to set the AWG. Note that not all rates are supported and a rate as close to the one you enter will be put.

        Args:
            rate (float) : target sample rate for the AWG (unit : Sa/s).
        """
        self._sample_rate = self.uploader.get_effective_sample_rate(rate)

        msg = f"effective sampling rate is set to {si_format(self._sample_rate, precision=1)}Sa/s"
        logging.info(msg)
        print("Info : " + msg)

    @property
    def repetition_alignment(self):
        return self._alignment

    @repetition_alignment.setter
    def repetition_alignment(self, value):
        self._alignment = value

    @property
    def measurements_description(self):
        return self._measurements_description

    def _get_measurement_converter(self):
        if self._measurement_converter is None:
            self._measurement_converter = MeasurementConverter(self._measurements_description,
                                                               self.n_rep, self._acquisition_conf.sample_rate)
        return self._measurement_converter

    def add_sequence(self, sequence):
        '''
        Adds a sequence to this object.
        Args:
            sequence (array) : array of segment_container
        '''
        # check input
        for entry in sequence:
            if isinstance(entry, segment_container) or isinstance(entry, conditional_segment):
                self.sequence.append(entry)
            else:
                raise ValueError('The provided element in the sequence seems to be of the wrong data type.'
                                 f'{type(entry)} provided, segment_container expected')

        for seg_container in self.sequence:
            if seg_container.sample_rate is not None:
                effective_rate = self.uploader.get_effective_sample_rate(seg_container.sample_rate)
                msg = f"effective sampling rate for {seg_container.name} is set to {si_format(effective_rate, precision=1)}Sa/s"
                logging.info(msg)
                print("Info : " + msg)

        # update dimensionality of all sequence objects
        for seg_container in self.sequence:
            seg_container.enter_rendering_mode()
            self._shape = find_common_dimension(seg_container.shape, self._shape)

        # Set the waveform cache equal to the sum over all channels and segments of the max axis length.
        # The cache will than be big enough for 1D iterations along every axis. This gives best performance
        total_axis_length = 0
        for seg_container in self.sequence:
            if not isinstance(seg_container, conditional_segment):
                for channel_name in seg_container.channels:
                    shape = seg_container[channel_name].data.shape
                    total_axis_length += max(shape)
            else:
                for branch in seg_container.branches:
                    for channel_name in branch.channels:
                        shape = branch[channel_name].data.shape
                        total_axis_length += max(shape)
        parent_data.set_waveform_cache_size(total_axis_length)

        self._shape = tuple(self._shape)
        self._sweep_index = [0]*self.ndim
        self._HVI_variables = data_container(marker_HVI_variable())
        self._HVI_variables = update_dimension(self._HVI_variables, self.shape)

        # enforce master clock for the current segments (affects the IQ channels (translated into a phase shift) and and the marker channels (time shifts))
        t_tot = np.zeros(self.shape)

        for seg_container in self.sequence:
            # NOTE: the time shift applies only to HVI markers.
            #       A segment with HVI markers can only be added once to the sequence.
            lp_time = loop_obj(no_setpoints=True)
            lp_time.add_data(t_tot, axis=list(range(self.ndim -1,-1,-1)))
            seg_container.add_master_clock(lp_time)
            self._HVI_variables += seg_container.software_markers.pulse_data_all
            if isinstance(seg_container, conditional_segment):
                self._check_conditional(seg_container, t_tot)
            self._measurements_description.add_segment(seg_container, t_tot)

            t_tot += seg_container.total_time
        self._measurements_description.add_HVI_variables(self._HVI_variables)

        self.params =[]

        for i in range(len(self.labels)):
            par_name = self.labels[i].replace(' ','_')
            set_param = index_param(par_name, self, dim = i)
            self.params.append(set_param)
            setattr(self, par_name, set_param)

        self._create_metadata()

    def _create_metadata(self):
        self.metadata = {}
        for (i,pc) in enumerate(self.sequence):
            md = pc.get_metadata()
            self.metadata[('pc%i'%i)] = md
        LOdict = {}
        for iq in self.sequence[0]._IQ_channel_objs:
            for vm in iq.qubit_channels:
                name = vm.channel_name
                LOdict[name] = iq.LO
        self.metadata['LOs'] = LOdict


    def _check_conditional(self, conditional:conditional_segment, total_time):

        if not getattr(self.uploader, 'supports_conditionals', False):
            raise Exception(f'Backend does not support conditional segments')

        condition = conditional.condition
        refs = condition if isinstance(condition, Iterable) else [condition]

        # Lookup acquistions for condition
        acquisition_names = self._get_acquisition_names(refs)
        logging.info(f'acquisitions: {acquisition_names}')

        # check start of conditional pulse
        min_slack = self._get_min_slack(acquisition_names, total_time)
        logging.info(f'min slack for conditional {min_slack} ns. (Must be < 0)')
        if min_slack < 0:
            raise Exception(f'condition triggered {-min_slack} ns too early')

        pass

    def _get_acquisition_names(self, refs:List[MeasurementRef]):
        acquisition_names = set()
        for ref in refs:
            acquisition_names.update(ref.keys)

        return list(acquisition_names)

    def _get_min_slack(self, acquisition_names, seg_start_times):
        # calculate slack for all sequence indices
        slack = np.empty((len(acquisition_names), ) + seg_start_times.shape)

        for i, name in enumerate(acquisition_names):
            slack[i] = seg_start_times - self._measurements_description.end_times[name]

        slack -= self.uploader.get_roundtrip_latency()

        return np.min(slack)

    def voltage_compensation(self, compensate):
        '''
        add a voltage compensation at the end of the sequence
        Args:
            compensate (bool) : compensate yes or no (default is True)
        '''
        self.neutralize = compensate

    def set_hw_schedule(self, hw_schedule: HardwareSchedule, **kwargs):
        '''
        Sets hardware schedule for the sequence.
        Args:
            hw_schedule: object with load() and start() methods to load and start the hardware schedule.
            kwargs : keyword arguments to be passed to the schedule.
        '''
        self.hw_schedule = hw_schedule
        self.hw_schedule.set_schedule_parameters(**kwargs)


    @property
    def configure_digitizer(self):
        return self._acquisition_conf.configure_digitizer

    @configure_digitizer.setter
    def configure_digitizer(self, enable):
        self._acquisition_conf.configure_digitizer = enable

    def set_acquisition(self,
                        t_measure=None,
                        sample_rate=None,
                        channels=[],
                        average_repetitions=None,
                        ):
        '''
        Args:
            t_measure (Union[float, loop_obj]):
                measurement time in ns. If None it must be specified in the acquire() call.
            sample_rate (float):
                Output data rate in Hz. When not None, the data should not be averaged,
                but sampled with specified rate. Useful for time traces and Elzerman readout.
                Does not change digitizer DAC rate. Data is down-sampled using block averages.
            average_repetitions (bool): Average data over the sequence repetitions.
        '''
        if self._measurement_converter is not None:
            raise Exception('Acquisition parameters cannot be changed after calling  '
                            'get_measurement_results or get_measurement_param')
        conf = self._acquisition_conf
        if t_measure:
            conf.t_measure = t_measure
            if sample_rate or conf.sample_rate:
                self._set_num_samples()
        if sample_rate:
            conf.sample_rate = sample_rate
            self._set_num_samples()
        if channels != []:
            conf.channels = channels
        if average_repetitions is not None:
            conf.average_repetitions = average_repetitions # @@@ implement Keysight

    def _set_num_samples(self):
        default_t_measure = self._acquisition_conf.t_measure
        sample_rate = self._acquisition_conf.sample_rate
        for m in self._measurements_description.measurements:
            if m.t_measure is None:
                if default_t_measure is None:
                    raise Exception(f't_measure not specified for measurement {m}')
                t_measure = default_t_measure
            elif isinstance(m.t_measure, Number):
                t_measure = m.t_measure
            else:
                raise Exception(f't_measure must be number and not a {type(m.t_measure)} for time traces')
            m.n_samples = self.uploader.get_num_samples(m.acquisition_channel,
                                                        t_measure, sample_rate) # @@@ implement QS, Tektronix


    def get_acquisition_param(self, name, upload=None, n_triggers=None): # @@@ remove
        if not self.configure_digitizer:
            raise Exception('configure_digitizer not set')
        if upload == 'auto':
            reader = SequencePlayer(self)
        else:
            reader = self

        conf = self._acquisition_conf
        acq_channels = conf.channels if conf.channels else list(self._digitizer_channels.keys())

        param = AcquisitionParam(reader, name,
                 acq_channels,
                 n_rep=self.n_rep if self.n_rep > 1 else None,
                 n_triggers=n_triggers,
                 t_measure=conf.t_measure,
                 sample_rate=conf.sample_rate,
                 average_repetitions=False)

        return param

    def get_measurement_param(self, name='seq_measurements', upload=None,
                              raw=True, states=True, values=True,
                              selectors=True, total_selected=True, accept_mask=True,
                              iq_complex=True):
        if not self.configure_digitizer:
            raise Exception('configure_digitizer not set')
        # @@@ 'always' vs 'auto'
        if upload == 'auto':
            reader = SequencePlayer(self)
        else:
            reader = self
        mc = self._get_measurement_converter()
        selection = DataSelection(raw=raw, states=states, values=values,
                                  selectors=selectors, total_selected=total_selected,
                                  accept_mask=accept_mask,
                                  iq_complex=iq_complex)
        param = MeasurementParameter(name, reader, mc, selection)
        return param

    def upload(self, index=None):
        '''
        Sends the sequence with the provided index to the uploader module. Once he is done, the play function can do its work.
        Args:
            index (tuple) : index if wich you wannt to upload. This index should fit into the shape of the sequence being played.

        Remark that upload and play can run at the same time and it is best to
        start multiple uploads at once (during upload you can do playback, when the first one is finihsed)
        (note that this is only possible if you AWG supports upload while doing playback)
        '''
        if index is None:
            index = self.sweep_index[::-1]
        self._validate_index(index)
        upload_job = self.uploader.create_job(self.sequence, index, self.id, self.n_rep, self._sample_rate,
                                              self.neutralize, alignment=self._alignment)
        upload_job.set_acquisition_conf(self._acquisition_conf)

        if self.hw_schedule is not None:
            upload_job.add_hw_schedule(self.hw_schedule, self._HVI_variables.item(tuple(index)).HVI_markers)

        self.uploader.add_upload_job(upload_job)

        return upload_job


    def play(self, index=None, release= True):
        '''
        Playback a certain index, assuming the index is provided.
        Args:
            index (tuple) : index if wich you wannt to upload. This index should fit into the shape of the sequence being played.
            release (bool) : release memory on the AWG after the element has been played.

        '''
        if index is None:
            index = self.sweep_index[::-1]
        self._validate_index(index)
        self.uploader.play(self.id, index, release)


    def plot(self, index=None, segments=None, awg_output=True):
        '''
        Plot sequence for specified index and segments.
        Args:
            index (tuple): index in sequence. If None use last index set via sweep params.
            segments (list[int]): indices of segments to plot. If None, plot all.
            awg_output (bool): if True plot output of AWGs, else plot virtual data.
        '''
        if index is None:
            index = self.sweep_index[::-1]

        if segments is None:
            segments = range(len(self.sequence))
        for s in segments:
            pt.figure()
            pt.title(f'Segment {s} index:{index}')
            self.sequence[s].plot(index, render_full=awg_output)

    def get_measurement_results(self, index=None, iq_complex=True):
        '''
        Returns data per measurement.
        Raw & state
        Raw = complex
        No averaging over repetitions.
        '''
        if index is None:
            index = self.sweep_index[::-1]
        mc = self._get_measurement_converter()
        mc.set_channel_data(self.get_channel_data(index))
        return mc.get_all_measurements(iq_complex=iq_complex)

    def get_measurement_data(self, index=None):
        '''
        Deprecated
        Returns channel data in V (!)
        '''
        logging.warning('get_measurement_data is deprecated. Use get_channel_data')
        return {
            name:value/1000.0
            for name, value in self.get_channel_data(index).items()
            }


    def get_channel_data(self, index=None):
        '''
        Returns acquisition data in mV per channel in a 1D or 2D array.
        The array is 1D for video mode scans and 2D for single shot measurements.
        Args:
            index: If None, use last played sequence index.
        '''
        if not self.configure_digitizer:
            raise Exception('configure_digitizer not set')
        return self.uploader.get_channel_data(self.id, index)

    def close(self):
        '''
        Closes the sequencer and releases all memory and resources. Sequencer cannot be used anymore.
        '''
        if self.hw_schedule:
            self.hw_schedule.stop()
            # NOTE: unloading the schedule is a BAD idea. If the next sequence uses the same schedule it costs ~ 1s to load it again.
            # self.hw_schedule.unload()
            self.hw_schedule = None
        if not self.sequence:
            return
        for seg_container in self.sequence:
            seg_container.exit_rendering_mode()
        self.sequence = None
        self.uploader.release_memory(self.id)

    def release_memory(self, index=None):
        '''
        function to free up memory in the AWG manually. By default the sequencer class will do garbarge collection for you (e.g. delete waveforms after playback)
        Args:
            index (tuple) : index if wich you want to release. If none release memory for all indexes.
        '''
        if index is not None:
            self._validate_index(index)
        self.uploader.release_memory(self.id, index)


    def set_sweep_index(self,dim,value):
        self._sweep_index[dim] = value

    def __del__(self):
        logging.debug(f'destructor seq: {self.id}')
        self.release_memory()

    def _validate_index(self, index):
        '''
        Raises an exception when the index is not valid.
        '''
        if len(index) != len(self._shape):
            raise Exception(f'Index {index} does not match sequence shape {self._shape}')
        if any(i >= s for i,s in zip(index, self._shape)):
            raise IndexError(f'Index {index} out of range; sequence shape {self._shape}')

class index_param(Parameter):
    def __init__(self, name, my_seq, dim):
        self.my_seq = my_seq
        self.dim = dim
        self.values = my_seq.setpoints[dim]
        val_map = dict(zip(self.values, range(len(self.values))))
        super().__init__(name=name, val_mapping = val_map, initial_value = self.values[0])

    def set_raw(self, value):
        self.my_seq.set_sweep_index(self.dim, value)


if __name__ == '__main__':
    import pulse_lib.segments.utility.looping as lp

    a = segment_container(["a", "b"])
    b = segment_container(["a", "b"])

    b.a.add_block(0,lp.linspace(30,100,10),100)
    b.a.reset_time()
    a.add_HVI_marker("marker_name", 20)
    b.add_HVI_marker("marker_name2", 50)

    b.add_HVI_variable("my_vatr", 800)
    a.a.add_block(20,lp.linspace(50,100,10, axis = 1, name = "time", unit = "ns"),100)

    b.slice_time(0,lp.linspace(80,100,10, name = "time", unit = "ns", axis= 2))

    my_seq = [a,b]

    seq = sequencer(None, dict())
    seq.add_sequence(my_seq)
    print(seq.HVI_variables.flat[0].HVI_markers)
    # print(seq.labels)
    # print(seq.units)
    # print(seq.setpoints)
    seq.upload([0])