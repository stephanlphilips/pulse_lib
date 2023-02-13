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
from .acquisition.player import SequencePlayer
from .acquisition.measurement_converter import MeasurementConverter, DataSelection, MeasurementParameter

from si_prefix import si_format

from typing import List
from collections.abc import Iterable
from numbers import Number
import numpy as np
import uuid
import logging

logger = logging.getLogger(__name__)

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
        '''
        Number times the sequence is repeated.
        If None or 0 the sequence is executed 1 time and the dimension 'repetition' will
        not be present in the measurement data.
        If n_rep is 1 then the dimension 'repetition' will be present in the measurement data.
        '''
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
        logger.info(msg)
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
            self._set_num_samples()
            self._measurements_description.calculate_measurement_offsets()
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
                logger.info(msg)

        # update dimensionality of all sequence objects
        logger.debug('Enter pre-rendering')
        for seg_container in self.sequence:
            seg_container.enter_rendering_mode()
            self._shape = find_common_dimension(self._shape, seg_container.shape)
        logger.debug('Done pre-render')
        # Set the waveform cache equal to the sum over all channels and segments of the max axis length.
        # The cache will than be big enough for 1D iterations along every axis. This gives best performance
        total_axis_length = 0
        n_samples = 0
        for seg_container in self.sequence:
            sr = seg_container.sample_rate if seg_container.sample_rate else 1e9
            n_samples = max(n_samples, np.max(seg_container.total_time) * 1e9 / sr)
            if not isinstance(seg_container, conditional_segment):
                for channel_name in seg_container.channels:
                    shape = seg_container[channel_name].data.shape
                    total_axis_length += max(shape)
            else:
                for branch in seg_container.branches:
                    for channel_name in branch.channels:
                        shape = branch[channel_name].data.shape
                        total_axis_length += max(shape)
        # limit cache to 8 GB
        max_cache = int(1e9 / n_samples)
        cache_size = min(total_axis_length, max_cache)
        logger.info(f'waveform cache: {cache_size} waveforms of max {n_samples} samples')
        parent_data.set_waveform_cache_size(cache_size)

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
            set_param = index_param(par_name, self.labels[i], self.units[i],
                                    self, dim = i)
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
        logger.info(f'acquisitions: {acquisition_names}')

        # check start of conditional pulse
        min_slack = self._get_min_slack(acquisition_names, total_time)
        logger.info(f'min slack for conditional {min_slack} ns. (Must be < 0)')
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
        if sample_rate:
            conf.sample_rate = sample_rate
        if channels != []:
            conf.channels = channels
        if average_repetitions is not None:
            conf.average_repetitions = average_repetitions # @@@ implement Keysight

    def _set_num_samples(self):
        default_t_measure = self._acquisition_conf.t_measure
        sample_rate = self._acquisition_conf.sample_rate
        for m in self._measurements_description.measurements:
            if m.n_repeat is not None:
                m.n_samples = m.n_repeat
                if sample_rate is not None:
                    logger.info(f'Ignoring sample_rate for measurement {m.name} because n_repeat is set')
            elif sample_rate is not None:
                if m.t_measure is None:
                    if default_t_measure is None:
                        raise Exception(f't_measure not specified for measurement {m}')
                    t_measure = default_t_measure
                elif isinstance(m.t_measure, Number):
                    t_measure = m.t_measure
                else:
                    raise Exception(f't_measure must be number and not a {type(m.t_measure)} for time traces')
                # @@@ implement QS, Tektronix
                if hasattr(self.uploader, 'actual_acquisition_points'):
                    m.n_samples, m.interval = self.uploader.actual_acquisition_points(m.acquisition_channel,
                                                                                      t_measure, sample_rate)
                else:
                    print(f'WARNING {type(self.uploader)} is missing method actual_acquisition_points(); using old computation')
                    m.n_samples = self.uploader.get_num_samples(
                            m.acquisition_channel, t_measure, sample_rate)
                    m.interval = round(1e9/sample_rate)
            else:
                m.n_samples = 1

    def get_measurement_param(self, name='seq_measurements', upload=None,
                              states=True, values=True,
                              selectors=True, total_selected=True, accept_mask=True,
                              iq_mode='Complex', iq_complex=None):
        '''
        Returns a qcodes MultiParameter with an entry per measurement, i.e. per acquire call.
        The data consists of raw data and derived data.
        The arguments of this method and the acquire call determine
        which entries are present in the parameter.

        For a call `acquire(start, t_measure, ref=name, threshold=threshold,
        accept_if=condition)`, the parameter can contain the
        following entries:
            "{name}":
                Raw data of the acquire call in mV.
                1D array with length n repetitions not a time trace.
                When sample_rate is set with set_acquisition(sample_rate=sr),
                then the data contains time traces in a 2D array indexed
                [index_repetition][time_step].
                Only present when channel contains no IQ data or
                when `iq_complex=True` or `iq_mode in['Complex','I','Q','amplitude','phase']`.
            "{name}_I":
                Similar to "{name}", but contains I component of IQ.
                Only present when channel contains IQ data and `iq_mode='I+Q'`.
            "{name}_Q":
                Similar to "{name}", but contains Q component of IQ.
                Only present when channel contains IQ data and `iq_mode='I+Q'`.
            "{name}_amp":
                Similar to "{name}", but contains amplitude of IQ.
                Only present when channel contains IQ data and `iq_mode='amplitude+phase'`.
            "{name}_phase":
                Similar to "{name}", but contains phase of IQ.
                Only present when channel contains IQ data and `iq_mode='amplitude+phase'`.
            "{name}_state":
                Qubit states in 1 D array.
                Only present when `states=True`, threshold is set,
                and accept_if is None.
            "{name}_frac":
                Fraction of qubit states == 1 in scalar value in range [0, 1].
                A value is only added to this average when all selectors (accept_if)
                have the required value.
                Only present when `values=True`, threshold is set,
                and accept_if is None.
            "{name}_selected":
                The qubit state of the measurements with an accept_if
                condition returned in a 1D array.
                Only present when `selectors=True`, threshold is set,
                and accept_if is set.
            "total_selected":
                The number of accepted sequence shots.
                A shot is accepted when all selectors have the required value.
                Only present when there is a least 1 measurement with
                accept_if condition set, and `total_selected=True`.
            "mask":
                A 1D array indicating per shot whether it is accepted (1) or
                rejected (0).
                Only present when there is a least 1 measurement with
                accept_if condition set, and `accept_mask=True`.

        Args:
            name (str): name of the qcodes parameter.
            upload (str):
                If 'auto' uploads, plays and retrieves data.
                Otherwise only retrieves data.
            states (bool): If True return the qubit state after applying threshold.
            values (bool): If True returns the fraction of qubits with state = |1>.
            selectors (bool):
                If True returns the qubit state of the measurements that
                have the argument `accept_if` defined in the acquire call.
            total_selected (bool):
                If True returns the number of accepted sequence shots.
                A shot is accepted when all selectors have the required value.
            accept_mask (bool):
                If True returns per shot whether it is accepted or not.
            iq_mode (str):
                when channel contains IQ data, i.e. iq_input=True or frequency is not None,
                then this parameter specifies how the complex I/Q value should be returned:
                    'Complex': return IQ data as complex value.
                    'I': return only I value.
                    'Q': return only Q value.
                    'amplitude': return amplitude.
                    'phase:' return phase [radians],
                    'I+Q', return I and Q using channel name postfixes '_I', '_Q'.
                    'amplitude+phase'. return amplitude and phase using channel name postfixes '_amp', '_phase'.
            iq_complex (bool):
                If False this is equivalent to `iq_mode='I+Q'`

        '''
        if not self.configure_digitizer:
            raise Exception('configure_digitizer not set')
        # @@@ 'always' vs 'auto'
        if upload == 'auto':
            reader = SequencePlayer(self)
        else:
            reader = self
        mc = self._get_measurement_converter()
        if iq_complex == False:
            iq_mode = 'I+Q'
        selection = DataSelection(raw=True, states=states, values=values,
                                  selectors=selectors, total_selected=total_selected,
                                  accept_mask=accept_mask,
                                  iq_mode=iq_mode)
        param = MeasurementParameter(name, reader, mc, selection)
        return param

    def _retry_upload(self, exception, n_retries, index):
        # '-8033' is a Keysight waveform upload error that requires a new upload
        if '(-8033)' in repr(exception) and n_retries > 0:
            logger.info('Upload failure', exc_info=True)
            logger.warning(f'Sequence upload failed at index {index}; retrying...')
            return True
        return False

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
        n_retries = 3
        while True:
            try:
                upload_job = self.uploader.create_job(self.sequence, index, self.id,
                                                      self.n_rep, self._sample_rate,
                                                      self.neutralize, alignment=self._alignment)
                upload_job.set_acquisition_conf(self._acquisition_conf)

                if self.hw_schedule is not None:
                    hvi_markers = self._HVI_variables.item(tuple(index)).HVI_markers
                    upload_job.add_hw_schedule(self.hw_schedule, hvi_markers)

                self.uploader.add_upload_job(upload_job)
                return upload_job
            except Exception as ex:
                if self._retry_upload(ex, n_retries, index):
                    n_retries -= 1
                else:
                    raise

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

        n_retries = 3
        while True:
            try:
                self.uploader.play(self.id, index, release)
                return
            except Exception as ex:
                if self._retry_upload(ex, n_retries, index):
                    n_retries -= 1
                    # Retries are only done for Keysight and require a new upload of the waveform
                    self.upload(index)
                else:
                    raise


    def plot(self, index=None, segments=None, awg_output=True, channels=None):
        '''
        Plot sequence for specified index and segments.
        Args:
            index (tuple): index in sequence. If None use last index set via sweep params.
            segments (list[int]): indices of segments to plot. If None, plot all.
            awg_output (bool): if True plot output of AWGs, else plot virtual data.
            channels (list[str]): names of channels to plot, if None, plot all.
        '''
        if index is None:
            index = self.sweep_index[::-1]

        if segments is None:
            segments = range(len(self.sequence))
        for s in segments:
            pt.figure()
            pt.title(f'Segment {s} index:{index}')
            self.sequence[s].plot(index, channels=channels, render_full=awg_output)

    def get_measurement_results(self, index=None,
                                raw=True, states=True, values=True,
                                selectors=True, total_selected=True,
                                accept_mask=True, iq_mode='Complex',
                                iq_complex=None):
        '''
        Returns data per measurement, i.e. per acquire call.
        The data consists of raw data and derived data.
        The arguments of this method and the acquire call determine
        which keys are present in the returned dictionary.

        For a call `acquire(start, t_measure, ref=name, threshold=threshold,
        accept_if=condition)`, the returned dictionary can contain the
        following entries:
            "{name}":
                Raw data of the acquire call in mV.
                1D array with length n repetitions not a time trace.
                When sample_rate is set with set_acquisition(sample_rate=sr),
                then the data contains time traces in a 2D array indexed
                [index_repetition][time_step].
                Only present when channel contains no IQ data or
                when `iq_complex=True` or `iq_mode in['Complex','I','Q','amplitude','phase']`.
            "{name}_I":
                Similar to "{name}", but contains I component of IQ.
                Only present when channel contains IQ data,
                `raw=True`, and `iq_mode='I+Q'`.
            "{name}_Q":
                Similar to "{name}", but contains Q component of IQ.
                Only present when channel contains IQ data,
                `raw=True`, and `iq_mode='I+Q'`.
            "{name}_amp":
                Similar to "{name}", but contains amplitude of IQ.
                Only present when channel contains IQ data,
                `raw=True`, and `iq_mode='amplitude+phase'`.
            "{name}_phase":
                Similar to "{name}", but contains phase of IQ.
                Only present when channel contains IQ data,
                `raw=True`, and ``iq_mode='amplitude+phase'`.
            "{name}_state":
                Qubit states in 1 D array.
                Only present when `states=True`, threshold is set,
                and accept_if is None.
            "{name}_frac":
                Fraction of qubit states == 1 in scalar value in range [0, 1].
                A value is only added to this average when all selectors (accept_if)
                have the required value.
                Only present when `values=True`, threshold is set,
                and accept_if is None.
            "{name}_selected":
                The qubit state of the measurements with an accept_if
                condition returned in a 1D array.
                Only present when `selectors=True`, threshold is set,
                and accept_if is set.
            "total_selected":
                The number of accepted sequence shots.
                A shot is accepted when all selectors have the required value.
                Only present when there is a least 1 measurement with
                accept_if condition set, and `total_selected=True`.
            "mask":
                A 1D array indicating per shot whether it is accepted (1) or
                rejected (0).
                Only present when there is a least 1 measurement with
                accept_if condition set, and `accept_mask=True`.

        Args:
            index (tuple[int, ..]):
                index in sequence when sweeping parameters. If None uses
                the index of the last play call.
            raw (bool):
                If True return raw measurement data.
            states (bool): If True return the qubit state after applying threshold.
            values (bool): If True returns the fraction of qubits with state = |1>.
            selectors (bool):
                If True returns the qubit state of the measurements that
                have the argument `accept_if` defined in the acquire call.
            total_selected (bool):
                If True returns the number of accepted sequence shots.
                A shot is accepted when all selectors have the required value.
            accept_mask (bool):
                If True returns per shot whether it is accepted or not.
            iq_mode (str):
                when channel contains IQ data, i.e. iq_input=True or frequency is not None,
                then this parameter specifies how the complex I/Q value should be returned:
                    'Complex': return IQ data as complex value.
                    'I': return only I value.
                    'Q': return only Q value.
                    'amplitude': return amplitude.
                    'phase:' return phase [radians],
                    'I+Q', return I and Q using channel name postfixes '_I', '_Q'.
                    'amplitude+phase'. return amplitude and phase using channel name postfixes '_amp', '_phase'.
            iq_complex (bool):
                If False this is equivalent to `iq_mode='I+Q'`

        '''
        if index is None:
            index = self.sweep_index[::-1]
        mc = self._get_measurement_converter()
        mc.set_channel_data(self.get_channel_data(index))
        if iq_complex == False:
            iq_mode = 'I+Q'
        selection = DataSelection(raw=raw, states=states, values=values,
                                  selectors=selectors, total_selected=total_selected,
                                  accept_mask=accept_mask,
                                  iq_mode=iq_mode)
        return mc.get_measurements(selection)


    def get_channel_data(self, index=None):
        '''
        Returns acquisition data in mV per channel in a 1D or 2D array, depending
        on the average_repetitions setting and n_rep. See set_acquisition().

        Video mode will generally use average_repetitions = True and thus return 1D data.

        The 2D data is arranged as [index_repetition][index_sample].
        The data of all acquire calls in a sequence is concatenated to one array.
        The methods get_measurement_result() and get_measurement_param() return
        the data per acquire call.

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
        logger.debug(f'destructor seq: {self.id}')
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
    def __init__(self, name, label, unit, my_seq, dim):
        self.my_seq = my_seq
        self.dim = dim
        self.values = my_seq.setpoints[dim]
        val_map = dict(zip(self.values, range(len(self.values))))
        super().__init__(
                name=name,
                label=label,
                unit=unit,
                val_mapping = val_map,
                initial_value = self.values[0])

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

    my_seq = [a,b]

    seq = sequencer(None, dict())
    seq.add_sequence(my_seq)
    print(seq.HVI_variables.flat[0].HVI_markers)
    # print(seq.labels)
    # print(seq.units)
    # print(seq.setpoints)
    seq.upload([0])