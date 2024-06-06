from dataclasses import dataclass, field
from typing import Tuple, List
from numbers import Number
import logging
import numpy as np
from pulse_lib.segments.segment_measurements import measurement_acquisition, measurement_expression
from pulse_lib.acquisition.iq_modes import iq_mode2func
from qcodes import MultiParameter

logger = logging.getLogger(__name__)


@dataclass
class SetpointsSingle:
    name: str
    label: str
    unit: str
    shape: Tuple[int] = field(default_factory=tuple)
    setpoints: Tuple[Tuple[float]] = field(default_factory=tuple)
    setpoint_names: Tuple[str] = field(default_factory=tuple)
    setpoint_labels: Tuple[str] = field(default_factory=tuple)
    setpoint_units: Tuple[str] = field(default_factory=tuple)

    def __post_init__(self):
        self.name = self.name.replace(' ', '_')

    def append(self, setpoint_values, setpoint_name, setpoint_label, setpoint_unit):
        values = (tuple(setpoint_values), )
        for d in self.shape[::-1]:
            values = (values*d, )
        setpoints = self.setpoints + values
        self.shape += (len(setpoint_values), )
        self.setpoints = setpoints
        self.setpoint_names += (setpoint_name, )
        self.setpoint_labels += (setpoint_label, )
        self.setpoint_units += (setpoint_unit, )

    def with_attributes(self, name=None, unit=None):
        label = self.label
        if name is None:
            name = self.name
        else:
            if label.startswith(self.name):
                label = name + label[len(self.name):]
        if unit is None:
            unit = self.unit
        return SetpointsSingle(name, label, unit,
                               self.shape,
                               self.setpoints,
                               self.setpoint_names,
                               self.setpoint_labels,
                               self.setpoint_units)


class SetpointsMulti:
    '''
    Pass to MultiParameter using __dict__ attribute. Example:
        spm = setpoints_multi()
        param = MultiParameter(..., **spm.__dict__)
    '''

    def __init__(self, sps_list: List[SetpointsSingle]):
        self.names = tuple(sps.name for sps in sps_list)
        self.labels = tuple(sps.label for sps in sps_list)
        self.units = tuple(sps.unit for sps in sps_list)
        self.shapes = tuple(sps.shape for sps in sps_list)
        self.setpoints = tuple(sps.setpoints for sps in sps_list)
        self.setpoint_names = tuple(sps.setpoint_names for sps in sps_list)
        self.setpoint_labels = tuple(sps.setpoint_labels for sps in sps_list)
        self.setpoint_units = tuple(sps.setpoint_units for sps in sps_list)


@dataclass
class DataSelection:
    raw: bool = False
    states: bool = False
    values: bool = False
    selectors: bool = False
    total_selected: bool = False
    accept_mask: bool = False
    iq_mode: str = 'Complex'


class MeasurementParameter(MultiParameter):
    def __init__(self, name, source, mc, data_selection):
        setpoints = mc.get_setpoints(data_selection)
        super().__init__(name, **setpoints.__dict__)
        self._source = source
        self._mc = mc
        self._data_selection = data_selection
        self._derived_params = {}

    def add_derived_param(self, name, func, label=None, unit='mV',
                          time_trace=False,
                          sample_rate=None,
                          setpoints=None, setpoint_units=None,
                          setpoint_labels=None, setpoint_names=None):
        '''
        Create a parameter that is derived from a trace (such as an
        average). Input of the function is the array of channels that
        would be returned from get_raw() without derived parameters.

        Args:
            name (str): name of the parameter
            func (Callable[[Dict[str,np.ndarray]], np.ndarray]): function
                calculating derived parameter
            label (Optional[str]): label for the parameter
            unit (str): unit for the parameter
            time_trace (bool): if True `func` returns a time trace with time dimension
            sample_rate (float): if not None specifies the sample rate of the time trace
            setpoints (Optional[np.ndarray]): setpoints
            setpoint_units (Optional[np.ndarray]): setpoint units
            setpoint_labels (Optional[np.ndarray]): setpoint labels
            setpoint_names (Optional[np.ndarray]): setpoint names
        '''
        if label is None:
            label = name

        # check the shape returned by the derived parameter
        dummy_data = {name: np.zeros(shape)
                      for name, shape in zip(self.names, self.shapes)}
        dp_shape = np.shape(func(dummy_data))
        n_dim = len(dp_shape)
        n_rep = self._mc.n_rep

        if setpoints is None:
            # Determine setpoints from dp_shape
            n_samples = None
            if time_trace:
                if n_dim == 0 or n_dim > 2:
                    raise Exception('Expecting dimensions (time,) or (n_rep, time), '
                                    f'but got {dp_shape}')
                add_repetitions = n_dim == 2
                n_samples = dp_shape[-1]
            else:
                if n_dim > 1:
                    raise Exception('Expecting dimensions (,) or (n_rep,), '
                                    f'but got {dp_shape}')
                add_repetitions = n_dim == 1

            if add_repetitions and dp_shape[0] != n_rep:
                raise Exception('Dimension mismatch for n-repetitions. '
                                f'Expected {n_rep}, got {dp_shape[0]}')

            setpoints = SetpointsSingle(name, label, unit)

            if add_repetitions:
                setpoints.append(np.arange(n_rep), 'repetition', 'repetition', '')
            if time_trace:
                if sample_rate is None:
                    sample_rate = self._mc._sample_rate
                period = 1e9/sample_rate
                time = tuple(np.arange(n_samples) * period)
                setpoints.append(time, 'time', 'time', 'ns')

            self.setpoints = self.setpoints + (setpoints.setpoints,)
            self.setpoint_names = self.setpoint_names + (setpoints.setpoint_names,)
            self.setpoint_labels = self.setpoint_labels + (setpoints.setpoint_labels,)
            self.setpoint_units = self.setpoint_units + (setpoints.setpoint_units,)
        else:
            if setpoint_labels and setpoint_names and setpoint_units:
                self.setpoints = self.setpoints + (setpoints,)
                self.setpoint_names = self.setpoint_names + (setpoint_names,)
                self.setpoint_labels = self.setpoint_labels + (setpoint_labels,)
                self.setpoint_units = self.setpoint_units + (setpoint_units,)
            else:
                raise Exception('Setpoint names, units and labels are also required when specifying setpoints')

        self._derived_params[name] = func
        self.names += (name,)
        self.shapes += (dp_shape,)
        self.units += (unit,)
        self.labels += (label,)

    def add_sensor_histogram(self, sensor, bins, range, accepted_only=False):
        '''
        Adds histograms for all measurements of sensor.

        Args:
            sensor (str): name of sensor in pulse-lib.
            bins (int): number of bins in histogram.
            range (Tuple[float,float]): upper and lower edge of histogram.
            accepted_only (bool): if True only count accepted shots
        '''
        for m in self._mc._description.measurements:
            if getattr(m, 'acquisition_channel', None) == sensor:
                self.add_measurement_histogram(m.name, bins, range,
                                               accepted_only=accepted_only)

    def add_measurement_histogram(self, m_name, bins, range, accepted_only=False):
        '''
        Adds histogram for specified measurement.

        Args:
            m_name (str): name of measurement in sequence.
            bins (int): number of bins in histogram.
            range (Tuple[float,float]): upper and lower edge of histogram.
            accepted_only (bool): if True only count accepted shots
        '''
        def _histogram(data):
            if accepted_only:
                if 'mask' not in data:
                    raise Exception('Cannot filter on accepted. Accept mask is not in data.')
                d = data[m_name][data['mask'].astype(bool)]
            else:
                d = data[m_name]
            return np.histogram(d, bins=binedges)[0]/d.shape[0]

        binedges = np.linspace(range[0], range[1], bins+1)
        bincenters = (binedges[1:] + binedges[:-1])/2
        setpoints = (tuple(bincenters),)
        setpoint_names = ('sensor_val',)
        self.add_derived_param(
                f'{m_name}_hist',
                _histogram,
                unit='',
                setpoints=setpoints,
                setpoint_units=('mV',),
                setpoint_names=setpoint_names,
                setpoint_labels=setpoint_names)

    def get_raw(self):
        data = self._source.get_channel_data()
        index = self._source.sweep_index[::-1]
        self._mc.set_channel_data(data, index)

        data = self._mc.get_measurement_data(self._data_selection)

        if len(self._derived_params) > 0:
            # TODO use custom dict that raise a more useful exception instead of KeyError.
            data_map = {name: values for name, values in zip(self.names, data)}
            for name, dp in self._derived_params.items():
                dp_data = dp(data_map)
                data.append(dp_data)
                data_map[name] = dp_data

        return data


class MeasurementConverter:
    ALLOWED_RELATIVE_THRESHOLD_DEVIATION = 0.01
    '''
    Allowed maximum deviation resulting is a difference between HW and SW thresholded data.
    Deviation is relative with respect to range of measured values.
    Warnings are raised when there is a difference in thresholded data on a raw
    value that is further from the threshold than specified allowed deviation.
    This warning indicates that the raw data has a small range with respect to the
    resolution of the hardware. It could also be due to a problem in the HW or SW.

    Note: On Qblox the error in the *rotated* data before thresholding is ~IQ_signal_level/2**11.
    So the range of the signal should be at least IQ_signal_level / 20.
    '''
    ALLOWED_FRACTION_THRESHOLD_DIFFERENCES = 0.002
    '''
    Allowed fraction of measurements that has a difference between HW and SW thresholded data.
    A warning is raised when more measurements have a different thresholded value.
    This warning indicates that there is too much data very close to the threshold.
    This warning indicates that the raw data has a small range with respect to the
    resolution of the hardware. It could also be due to a problem in the HW or SW.
    '''

    def __init__(self, description, n_rep, sample_rate):
        self._description = description
        self.n_rep = n_rep
        self._sample_rate = sample_rate
        self._channel_raw = {}

        self._raw = []
        self._raw_is_iq = []
        self._states = []
        self._selectors = []
        self._values = []
        self._total_selected = []
        self._accepted = []

        self.sp_raw = []
        self.sp_states = []
        self.sp_selectors = []
        self.sp_values = []
        self.sp_total = []
        self.sp_mask = []
        self._generate_setpoints_raw()
        self._generate_setpoints()

    def _generate_setpoints_raw(self):
        n_rep = self.n_rep
        digitizer_channels = self._description.digitizer_channels
        for m in self._description.measurements:
            if not isinstance(m, measurement_acquisition):
                continue
            channel_name = m.acquisition_channel
            name = f'{m.name}'
            label = f'{m.name} ({channel_name}:{m.index})'

            sp_raw = SetpointsSingle(name, label, 'mV')
            if n_rep:
                sp_raw.append(np.arange(n_rep), 'repetition', 'repetition', '')
            if m.interval is not None and m.aggregate_func is None:
                n_samples = m.n_samples
                if not isinstance(n_samples, Number):
                    n_samples = max(n_samples)
                if m.f_sweep is None:
                    time = tuple(np.arange(n_samples, dtype=float) * m.interval)
                    sp_raw.append(time, 'time', 'time', 'ns')
                else:
                    f = tuple(np.linspace(m.f_sweep[0], m.f_sweep[1], n_samples, dtype=float))
                    sp_raw.append(f, 'frequency', 'frequency', 'Hz')

            self.sp_raw.append(sp_raw)
            channel = digitizer_channels[channel_name]
            self._raw_is_iq.append(channel.iq_out)

    def _generate_setpoints(self):
        for m in self._description.measurements:
            if isinstance(m, measurement_acquisition):
                if not m.has_threshold:
                    # do not add to result
                    continue
                if m.interval is not None and m.aggregate_func is None:
                    raise Exception(f'State threshold cannot be applied on time trace ({m.name})')

            name = f'{m.name}_state'
            label = name
            sp_state = SetpointsSingle(name, label, '')

            if self.n_rep:
                sp_state.append(np.arange(self.n_rep), 'repetition', 'repetition', '')
            self.sp_states.append(sp_state)

            if m.accept_if is not None:
                name = f'{m.name}_selected'
                label = name
                sp_result = SetpointsSingle(name, label, '#')
                self.sp_selectors.append(sp_result)
            else:
                name = f'{m.name}_fraction'
                label = name
                sp_result = SetpointsSingle(name, label, '')
                self.sp_values.append(sp_result)

        if len(self.sp_selectors) > 0:
            sp_mask = SetpointsSingle('mask', 'mask', '')
            if self.n_rep:
                sp_mask.append(np.arange(self.n_rep), 'repetition', 'repetition', '')
            self.sp_mask.append(sp_mask)
            self.sp_total.append(SetpointsSingle('total_selected', 'total_selected', '#'))

    def _get_names(self, selection):
        setpoints = self.get_setpoints(selection)
        return setpoints.names

    def _set_data_raw(self, index):
        self._raw = []
        self._hw_thresholded = {}
        for m in self._description.measurements:
            if isinstance(m, measurement_acquisition):
                channel_name = m.acquisition_channel
                channel_data = self._channel_raw[channel_name]
                data_offset = m.data_offset
                if not isinstance(data_offset, Number):
                    data_offset = data_offset[tuple(index)]
                if m.interval is None:
                    channel_raw = channel_data[..., data_offset]
                    thresholded = self._channel_raw.get(m.acquisition_channel+'.thresholded', None)
                    if thresholded is not None:
                        self._hw_thresholded[len(self._raw)] = thresholded[..., data_offset]
                else:
                    n_samples = m.n_samples
                    if not isinstance(n_samples, Number):
                        # NOTE: n_samples is an array (loop_obj)
                        shape = channel_data.shape[:-1]+(max(n_samples),)
                        channel_raw = np.full(shape, np.nan)
                        n_samples = n_samples[tuple(index)]
                        channel_raw[..., :n_samples] = channel_data[..., data_offset:data_offset+n_samples]
                    else:
                        channel_raw = channel_data[..., data_offset:data_offset+n_samples]
                    if m.aggregate_func:
                        t_start = 0  # TODO @@@ get from measurement
                        # aggregate time series
                        channel_raw = m.aggregate_func(t_start, channel_raw)
                self._raw.append(channel_raw)

    def _threshold_data(self, m, raw_index):
        values = self._raw[raw_index].real
        result = values > m.threshold
        if m.zero_on_high:
            result = result ^ 1
        result = result.astype(int)

        hw_thresholded = self._hw_thresholded.get(raw_index, None)
        if hw_thresholded is not None:
            print()
            print(f'Above threshold {np.sum(hw_thresholded)}, {np.sum(hw_thresholded)/len(values):%}')
        if hw_thresholded is not None and np.any(result != hw_thresholded):
            different = (result != hw_thresholded).nonzero()[0]
            n_different = len(different)
            max_value, min_value = np.max(values), np.min(values)
            value_differences = values[different] - m.threshold
            rel_differences = value_differences / (max_value - min_value)
            msg = (f"{n_different} differences between hardware and software thresholded results for '{m.name}'. "
                   f"Raw value range: [{min_value:.6f}, {max_value:.6f}] mV, max difference: {np.max(np.abs(value_differences)):.6f} mV")
            if (np.max(np.abs(rel_differences)) > MeasurementConverter.ALLOWED_RELATIVE_THRESHOLD_DEVIATION
                    or n_different > max(1, len(values)*MeasurementConverter.ALLOWED_FRACTION_THRESHOLD_DIFFERENCES)):
                logger.warning(msg)
                # level='WARNING'
            else:
                logger.info(msg)
                # level='INFO'
            logger.info(f"indices: {different}, values-threshold: {value_differences} mV")
            # print(level, msg)
            # print(f"{min(value_differences):.6f}, {max(value_differences):.6f}, {np.max(np.abs(rel_differences)):.2%}, {max_value-min_value:.6f}")

        return result

    def _set_states(self):
        # iterate through measurements and keep last named values in dictionary
        results = []
        selectors = []
        values_unfiltered = []
        last_result = {}
        n_rep = self.n_rep if self.n_rep else 1
        accepted_mask = np.ones(n_rep, dtype=int)
        acquisition_cnt = 0
        for m in self._description.measurements:
            if isinstance(m, measurement_acquisition):
                if not m.has_threshold:
                    acquisition_cnt += 1
                    # do not add to result
                    continue
                result = self._threshold_data(m, acquisition_cnt)
                acquisition_cnt += 1
            elif isinstance(m, measurement_expression):
                result = m.expression.evaluate(last_result)
            else:
                raise NotImplementedError(f'Unknown measurement type {type(m)}')

            if m.accept_if is not None:
                accepted = result == m.accept_if
                accepted_mask *= accepted
                selectors.append(np.sum(accepted))
            else:
                values_unfiltered.append(result)
            last_result[m.name] = result
            results.append(result)

        total_selected = np.sum(accepted_mask)
        self._states = results
        if len(selectors) > 0:
            self._accepted = [accepted_mask]
            self._total_selected = [total_selected]
        self._selectors = selectors
        if total_selected > 0:
            self._values = [np.sum(result*accepted_mask)/total_selected for result in values_unfiltered]
        else:
            logger.warning('No shot is accepted')
            self._values = [np.nan for result in values_unfiltered]

    def set_channel_data(self, data, index):
        self._channel_raw = data
        self._set_data_raw(index)
        self._set_states()

    def get_setpoints(self, selection):
        sp_list = []
        if selection.raw:
            for sp, is_iq in zip(self.sp_raw, self._raw_is_iq):
                if not is_iq:
                    sp_list.append(sp)
                else:
                    funcs = iq_mode2func(selection.iq_mode)
                    for postfix, _, unit in funcs:
                        sp_new = sp.with_attributes(name=sp.name+postfix, unit=unit)
                        sp_list.append(sp_new)
        if selection.states:
            sp_list += self.sp_states
        if selection.values:
            sp_list += self.sp_values
        if selection.selectors:
            sp_list += self.sp_selectors
        if selection.total_selected:
            sp_list += self.sp_total
        if selection.accept_mask:
            sp_list += self.sp_mask
        return SetpointsMulti(sp_list)

    def get_measurement_data(self, selection):
        data = []
        if selection.raw:
            for raw, is_iq in zip(self._raw, self._raw_is_iq):
                if not is_iq:
                    data.append(raw)
                else:
                    funcs = iq_mode2func(selection.iq_mode)
                    for _, func, _ in funcs:
                        data.append(func(raw))
        if selection.states:
            data += self._states
        if selection.values:
            data += self._values
        if selection.selectors:
            data += self._selectors
        if selection.total_selected:
            data += self._total_selected
        if selection.accept_mask:
            data += self._accepted
        return data

    def get_measurements(self, selection):
        result = {}
        for name, value in zip(self._get_names(selection), self.get_measurement_data(selection)):
            result[name] = value
        return result
